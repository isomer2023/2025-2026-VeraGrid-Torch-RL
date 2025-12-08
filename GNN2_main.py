# GNN2_main.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import simbench as sb
from torch_geometric.data import Data, Batch
import warnings
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress Pandas warnings
warnings.filterwarnings('ignore')

# Import custom modules for grid simulation and GNN model
try:
    import src.GNNDIF.GC_PandaPowerImporter as GCPPI
    from VeraGridEngine import api as gce
    from src.GNNDIF.gnnmodel import GridGNN
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    exit()

# ================= Configuration Parameters =================
SB_CODE = "1-MV-urban--0-sw"  # SimBench test case identifier
LR = 0.0005  # Learning rate
EPOCHS = 3000  # Total training epochs
BATCH_SIZE = 32  # Batch size for stable BatchNorm
HIDDEN_DIM = 128  # GNN hidden dimension
HEADS = 4  # Number of attention heads
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = "best_gnn_model.pth"  # Path to save the best model


# =============================================================
def get_gc_id(obj):
    """Extract identifier from GridCal object using multiple possible attributes."""
    if hasattr(obj, 'id') and obj.id is not None: return obj.id
    if hasattr(obj, 'idtag') and obj.idtag is not None: return obj.idtag
    if hasattr(obj, 'uuid') and obj.uuid is not None: return obj.uuid
    if hasattr(obj, 'name') and obj.name is not None: return obj.name
    return str(obj)


def _set_val(obj, attr_list, val):
    """Set value to object using first successful attribute from list."""
    for attr in attr_list:
        try:
            setattr(obj, attr, val)
            return
        except:
            continue


def _get_val(obj, attr_list, default=0.0):
    """Get value from object using first successful attribute from list."""
    for attr in attr_list:
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except:
                continue
    return default


def _is_slack_safe(g):
    """Determine if generator is a slack/reference bus."""
    val = getattr(g, 'is_slack', None)
    if val is not None: return bool(val)
    val = getattr(g, 'slack', None)
    if val is not None: return bool(val)
    name = str(getattr(g, 'name', ''))
    if "Ext_Grid" in name: return True
    return False


def get_safe_gen_results(driver, grid):
    """Extract generator active power results from OPF driver."""
    if driver and hasattr(driver, 'results'):
        res = driver.results
        if hasattr(res, 'gen_p') and res.gen_p is not None: return res.gen_p
        if hasattr(res, 'P') and res.P is not None: return res.P
    # Fallback: extract directly from generator objects
    p_list = []
    for g in grid.generators:
        p_val = _get_val(g, ['P', 'p_mw', 'p'], 0.0)
        p_list.append(p_val)
    return p_list


def setup_and_run_opf_teacher(grid, current_profile_sgen):
    """
    Teacher module: Solve full AC-OPF problem using GridCal engine.

    Args:
        grid: GridCal grid object
        current_profile_sgen: Dictionary of available renewable generation per unit

    Returns:
        OPF driver with optimization results
    """
    # Configure generator constraints and costs
    for g in grid.generators:
        g_name = str(getattr(g, 'name', ''))

        # Slack bus configuration (unlimited capacity, high cost)
        if _is_slack_safe(g) or "Ext_Grid" in g_name:
            _set_val(g, ['Pmax', 'P_max'], 99999.0)
            _set_val(g, ['Pmin', 'P_min'], -99999.0)
            _set_val(g, ['cost_a', 'Cost1'], 1.0)  # Quadratic cost coefficient
            _set_val(g, ['cost_b', 'Cost2'], 100.0)  # Linear cost coefficient
            _set_val(g, ['is_controlled', 'controlled'], True)

        # Renewable generator configuration (limited by available power)
        elif "sgen" in g_name:
            try:
                sgen_idx = int(g_name.split('_')[1])
                p_avail = current_profile_sgen.get(sgen_idx, 0.0)
            except:
                p_avail = 0.0

            # Set constraints: can generate between 0 and available power
            _set_val(g, ['Pmax', 'P_max'], p_avail)
            _set_val(g, ['Pmin', 'P_min'], 0.0)
            _set_val(g, ['cost_a', 'Cost1'], 0.01)  # Low cost for renewables
            _set_val(g, ['cost_b', 'Cost2'], 0.1)
            _set_val(g, ['is_controlled', 'controlled'], True)
            _set_val(g, ['Qmax', 'Q_max'], 0.0)  # No reactive power capability
            _set_val(g, ['Qmin', 'Q_min'], 0.0)

    # Configure OPF solver options
    opf_opts = gce.OptimalPowerFlowOptions()
    if hasattr(gce, 'SolverType'):
        _set_val(opf_opts, ['solver', 'solver_type'], gce.SolverType.NONLINEAR_OPF)

    opf_opts.objective = 0  # Minimize generation cost
    _set_val(opf_opts, ['activate_voltage_limits', 'voltage_limits'], True)
    _set_val(opf_opts, ['vmin', 'Vmin'], 0.98)  # Lower voltage limit (pu)
    _set_val(opf_opts, ['vmax', 'Vmax'], 1.02)  # Upper voltage limit (pu)
    _set_val(opf_opts, ['activate_thermal_limits', 'thermal_limits'], True)
    _set_val(opf_opts, ['dispatch_P', 'control_active_power'], True)
    _set_val(opf_opts, ['dispatch_Q', 'control_reactive_power'], False)
    _set_val(opf_opts, ['allow_soft_limits', 'soft_limits'], True)
    _set_val(opf_opts, ['initialize_with_dc', 'init_dc'], False)

    # Run OPF solver
    opf_driver = gce.OptimalPowerFlowDriver(grid, opf_opts)
    try:
        opf_driver.run()
    except Exception as e:
        print(f"⚠️ OPF solver failed: {e}")
    return opf_driver



def get_graph_data(grid, pf_results, bus_idx_map):
    """
    Convert power grid to PyTorch Geometric Data object for GNN processing.

    Node features (6-dim):
        0: Total active load at bus (scaled ×3)
        1: Total reactive load at bus (scaled ×3)
        2: Available renewable generation at bus (scaled /10)
        3: Voltage magnitude deviation from 1.0 pu (scaled ×10)
        4: Renewable generator flag (1 if bus has renewable, 0 otherwise)
        5: Voltage phase angle (radians)

    Edge features (4-dim):
        0: Line resistance R (scaled ×100)
        1: Line reactance X (scaled ×100)
        2: Line thermal rating (scaled /1000)
        3: Line loading percentage (absolute value)

    Returns:
        Data: PyG Data object with node/edge features
        sgen_mask: Boolean mask indicating renewable generator buses
    """
    num_nodes = len(grid.buses)
    x = np.zeros((num_nodes, 6), dtype=np.float32)

    # 1. Aggregate load at each bus
    for l in grid.loads:
        bus_ref = getattr(l, 'bus', getattr(l, 'node', None))
        if bus_ref:
            idx = bus_idx_map.get(get_gc_id(bus_ref))
            if idx is not None:
                p_val = _get_val(l, ['P', 'p_mw', 'p'])
                q_val = _get_val(l, ['Q', 'q_mvar', 'q'])
                x[idx, 0] += p_val * 3.0  # Scale active load
                x[idx, 1] += q_val * 3.0  # Scale reactive load

    # 2. Add generator information
    sgen_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for g in grid.generators:
        bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
        if bus_ref:
            idx = bus_idx_map.get(get_gc_id(bus_ref))
            if idx is not None:
                x[idx, 4] = 1.0  # Mark as generator bus
                if "sgen" in getattr(g, 'name', ''):
                    p_max = _get_val(g, ['Pmax', 'P_max'], 0.0)
                    x[idx, 2] += p_max / 10.0  # Scale available generation
                    sgen_mask[idx] = True

    # 3. Add voltage information from power flow results
    v_mag_scaled = np.zeros(num_nodes)
    v_ang = np.zeros(num_nodes)
    if pf_results and hasattr(pf_results, 'voltage'):
        v_complex = np.array(pf_results.voltage, dtype=np.complex128)
        if len(v_complex) == num_nodes:
            v_abs = np.abs(v_complex)
            v_mag_scaled = (v_abs - 1.0) * 10.0  # Deviation from nominal voltage
            v_ang = np.angle(v_complex)  # Phase angle in radians

    x[:, 3] = v_mag_scaled
    x[:, 5] = v_ang

    # 4. Build edge features for all branches (lines/transformers)
    src, dst, attr = [], [], []
    branches = []

    # Get all branch objects from grid
    if hasattr(grid, 'branches'):
        branches = grid.branches
    elif hasattr(grid, 'get_branches'):
        branches = grid.get_branches()
    else:
        branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))

    # Get line loading from power flow results
    branch_loadings = []
    if pf_results and hasattr(pf_results, 'loading'):
        branch_loadings = pf_results.loading

    # Process each branch
    for i, br in enumerate(branches):
        try:
            # Skip inactive branches
            if _get_val(br, ['active', 'status'], 1.0) < 0.5:
                continue

            # Get from and to bus references
            f_ref = getattr(br, 'bus_from', getattr(br, 'from_node', getattr(br, 'busFrom', None)))
            t_ref = getattr(br, 'bus_to', getattr(br, 'to_node', getattr(br, 'busTo', None)))

            if f_ref and t_ref:
                u = bus_idx_map.get(get_gc_id(f_ref))
                v = bus_idx_map.get(get_gc_id(t_ref))
                if u is not None and v is not None:
                    # Extract line parameters
                    r = float(_get_val(br, ['r', 'R']))
                    x_val = float(_get_val(br, ['x', 'X']))
                    rate = float(_get_val(br, ['rate', 'Rate'], 100.0))

                    # Get loading percentage
                    loading_val = 0.0
                    if branch_loadings is not None and i < len(branch_loadings):
                        loading_val = float(branch_loadings[i])

                    # Create edge features with appropriate scaling
                    feat_rate = rate / 1000.0  # Normalize thermal rating
                    feat_load = abs(loading_val)  # Absolute loading percentage
                    feat_r = r * 100.0  # Scale resistance
                    feat_x = x_val * 100.0  # Scale reactance

                    edge_feat = [feat_r, feat_x, feat_rate, feat_load]

                    # Add both directions (undirected graph)
                    src.extend([u, v])
                    dst.extend([v, u])
                    attr.extend([edge_feat, edge_feat])
        except Exception as e:
            continue

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
    edge_attr = torch.tensor(attr, dtype=torch.float32).to(DEVICE)

    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr), sgen_mask.to(DEVICE)

# --- Core Modules ---
def evaluate_model(model, grid, bus_idx_map, test_idx, df_load_p, df_load_q, df_sgen_p, device):
    """
    Comprehensive evaluation of trained GNN model on test dataset.

    Evaluates model performance by comparing predicted generation dispatch
    against optimal solutions from OPF teacher.

    Outputs:
        - CSV file with detailed results
        - Text file with evaluation metrics
        - Three visualization plots
    """
    print("\n" + "=" * 40)
    print("🧪 Model Evaluation on Test Set")
    print("=" * 40)

    # Load best saved model weights
    try:
        model.load_state_dict(torch.load("best_gnn_model.pth"))
        print("✅ Loaded best model weights: best_gnn_model.pth")
    except Exception as e:
        print(f"⚠️ Could not load model weights, using current weights: {e}")

    model.eval()

    # Randomly sample from test set (max 1000 samples)
    num_samples = 1000
    if len(test_idx) > num_samples:
        eval_indices = np.random.choice(test_idx, num_samples, replace=False)
    else:
        eval_indices = test_idx

    print(f"📊 Sample count: {len(eval_indices)} (from test set)")
    results_list = []

    with torch.no_grad():
        for i, t in enumerate(eval_indices):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(eval_indices)}...")

            # --- A. Reconstruct operating scenario ---
            stress_factor = np.random.uniform(6.0, 8.0)

            # Set loads from profiles
            current_load_p = df_load_p.iloc[t]
            current_load_q = df_load_q.iloc[t]
            for l in grid.loads:
                try:
                    idx = int(l.name.split('_')[1])
                    _set_val(l, ['P', 'p'], current_load_p.get(idx, 0.0))
                    _set_val(l, ['Q', 'q'], current_load_q.get(idx, 0.0))
                except:
                    pass

            # Set renewable generation with stress factor
            sgen_p_dict = (df_sgen_p.iloc[t] * stress_factor).to_dict()
            grid_pf = deepcopy(grid)
            for g in grid_pf.generators:
                if "sgen" in getattr(g, 'name', ''):
                    try:
                        idx = int(g.name.split('_')[1])
                        p_avail = sgen_p_dict.get(idx, 0.0)
                        _set_val(g, ['Pmax', 'P_max'], p_avail)
                        _set_val(g, ['P', 'p'], p_avail)
                    except:
                        pass

            # --- B. Compute base power flow ---
            pf_driver = None
            current_pf_results = None
            try:
                pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
                pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
                pf_driver.run()
                current_pf_results = pf_driver.results
            except:
                pass

            # --- C. Compute optimal dispatch (Teacher) ---
            teacher_driver = setup_and_run_opf_teacher(grid, sgen_p_dict)
            if not (teacher_driver and hasattr(teacher_driver.results, 'converged')
                    and teacher_driver.results.converged):
                continue

            gen_p_vec = get_safe_gen_results(teacher_driver, grid)

            # --- D. GNN prediction (Student) ---
            data, mask = get_graph_data(grid, current_pf_results, bus_idx_map)
            pred = model(data)

            # Compare predictions with optimal values
            for g_idx, g in enumerate(grid.generators):
                if "sgen" in getattr(g, 'name', ''):
                    bus = getattr(g, 'bus', getattr(g, 'node', None))
                    if bus:
                        node_idx = bus_idx_map.get(get_gc_id(bus))
                        if node_idx is not None:
                            # 1. Ground truth from OPF
                            p_opt = float(gen_p_vec[g_idx])
                            p_avail = _get_val(g, ['Pmax', 'P_max'])

                            # 2. Prediction from GNN
                            pred_val = float(pred[node_idx].item())

                            # 3. Filter and record (only generators with available capacity)
                            if p_avail > 0.001:
                                true_alpha = np.clip(p_opt / p_avail, 0.0, 1.0)
                                pred_alpha_clamped = np.clip(pred_val, 0.0, 1.0)

                                results_list.append({
                                    "Time_Step": t,
                                    "Gen_ID": g.name,
                                    "True_Alpha": true_alpha,
                                    "Pred_Alpha": pred_alpha_clamped,
                                    "Error": pred_alpha_clamped - true_alpha,
                                    "Abs_Error": abs(pred_alpha_clamped - true_alpha)
                                })

    # Create results DataFrame
    df = pd.DataFrame(results_list)
    if df.empty:
        print("❌ No valid test data collected.")
        return

    # Save detailed results
    df.to_csv("eval_results_detailed.csv", index=False)
    print(f"\n✅ Detailed results saved: eval_results_detailed.csv ({len(df)} records)")

    # Calculate evaluation metrics
    mae = mean_absolute_error(df["True_Alpha"], df["Pred_Alpha"])
    rmse = np.sqrt(mean_squared_error(df["True_Alpha"], df["Pred_Alpha"]))
    r2 = r2_score(df["True_Alpha"], df["Pred_Alpha"])

    print(f"\n🏆 Evaluation Metrics:")
    print(f"   MAE  : {mae:.6f}")
    print(f"   RMSE : {rmse:.6f}")
    print(f"   R2   : {r2:.6f}")

    # Save metrics to file
    with open("eval_metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\nSamples: {len(df)}")

    # Visualization
    sns.set_theme(style="whitegrid")

    # 1. Scatter plot: Predicted vs True values
    plt.figure(figsize=(8, 8))
    plt.scatter(df["True_Alpha"], df["Pred_Alpha"], alpha=0.15, s=10, color="#1f77b4")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Ideal")
    plt.xlabel("Teacher (OPF) Alpha")
    plt.ylabel("Student (GNN) Alpha")
    plt.title(f"Prediction vs Ground Truth (N={len(df)})\nMAE={mae:.4f}, R2={r2:.4f}")
    plt.legend()
    plt.savefig("eval_1_scatter.png", dpi=300)
    plt.close()

    # 2. Error distribution histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Error"], bins=100, kde=True, color="purple", stat="density")
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel("Error (Pred - True)")
    plt.title("Error Distribution Histogram")
    plt.savefig("eval_2_error_hist.png", dpi=300)
    plt.close()

    # 3. Boxplot of errors per generator
    plt.figure(figsize=(14, 6))
    gen_errors = df.groupby("Gen_ID")["Abs_Error"].mean().sort_values(ascending=False)
    top_gens = gen_errors.head(30).index
    df_filtered = df[df["Gen_ID"].isin(top_gens)]
    sns.boxplot(x="Gen_ID", y="Abs_Error", data=df_filtered, palette="Reds_r", order=top_gens)
    plt.xticks(rotation=90)
    plt.title("Absolute Error per Generator (Top 30 Worst Controlled)")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig("eval_3_gen_boxplot.png", dpi=300)
    plt.close()

    print("🖼️  All plots generated: eval_1_scatter.png, eval_2_error_hist.png, eval_3_gen_boxplot.png")


def main():
    """Main training pipeline for GNN-based OPF approximation."""
    print(f"🚀 Starting training: {SB_CODE} (Batch={BATCH_SIZE})")

    # 1. Prepare data and environment
    net_pp = sb.get_simbench_net(SB_CODE)
    grid = GCPPI.PP2GC(net_pp)
    bus_idx_map = {get_gc_id(b): i for i, b in enumerate(grid.buses)}

    print("📦 Loading profiles...")
    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_load_q = profiles[('load', 'q_mvar')]
    df_sgen_p = profiles[('sgen', 'p_mw')]

    # Split into training and testing sets
    n_time_steps = len(df_load_p)
    all_idx = np.arange(n_time_steps)
    split1 = int(0.8 * n_time_steps)
    train_idx = all_idx[:split1]
    test_idx = all_idx[split1:]

    # 2. Initialize model and optimizer
    model = GridGNN(num_node_features=6, num_edge_features=4,
                    hidden_dim=HIDDEN_DIM, heads=HEADS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n{'Epoch':<6} | {'Loss':<10} | {'Info'}")
    print("-" * 50)

    # Setup logging
    log_f = open("loss_log.csv", mode="w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["epoch", "time_index", "loss"])

    best_loss = float('inf')
    batch_data_list = []  # Temporary storage for batch accumulation

    optimizer.zero_grad()  # Initialize gradients

    # 3. Main training loop
    for epoch in range(EPOCHS):
        # Randomly select a time step from training set
        t = int(np.random.choice(train_idx))

        # --- Scenario generation ---
        stress_factor = np.random.uniform(6.0, 8.0)

        # Set loads from profiles
        current_load_p = df_load_p.iloc[t]
        current_load_q = df_load_q.iloc[t]
        for l in grid.loads:
            try:
                idx = int(l.name.split('_')[1])
                _set_val(l, ['P', 'p'], current_load_p.get(idx, 0.0))
                _set_val(l, ['Q', 'q'], current_load_q.get(idx, 0.0))
            except:
                pass

        # Set renewable generation with stress factor
        sgen_p_dict = (df_sgen_p.iloc[t] * stress_factor).to_dict()
        grid_pf = deepcopy(grid)
        for g in grid_pf.generators:
            if "sgen" in getattr(g, 'name', ''):
                try:
                    idx = int(g.name.split('_')[1])
                    p_avail = sgen_p_dict.get(idx, 0.0)
                    _set_val(g, ['Pmax', 'P_max'], p_avail)
                    _set_val(g, ['P', 'p'], p_avail)
                except:
                    pass

        # --- Base power flow calculation ---
        pf_driver = None
        current_pf_results = None
        try:
            pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
            pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
            pf_driver.run()
            current_pf_results = pf_driver.results
        except:
            pass

        # --- Teacher: Solve OPF for optimal dispatch ---
        teacher_driver = setup_and_run_opf_teacher(grid, sgen_p_dict)
        if not (teacher_driver and hasattr(teacher_driver.results, 'converged')
                and teacher_driver.results.converged):
            continue

        gen_p_vec = get_safe_gen_results(teacher_driver, grid)

        # --- Prepare target values for GNN ---
        # Create target vector: optimal generation ratio α for each bus
        full_target = torch.zeros(len(grid.buses), 1)
        valid_sample = False

        for i, g in enumerate(grid.generators):
            if "sgen" in getattr(g, 'name', ''):
                bus = getattr(g, 'bus', getattr(g, 'node', None))
                if bus:
                    idx = bus_idx_map.get(get_gc_id(bus))
                    if idx is not None:
                        p_opt = float(gen_p_vec[i])
                        p_avail = _get_val(g, ['Pmax', 'P_max'])
                        if p_avail > 0.001:
                            alpha = np.clip(p_opt / p_avail, 0.0, 1.0)
                            full_target[idx] = alpha
                            valid_sample = True

        if not valid_sample:
            continue

        # --- Get graph representation ---
        data, mask = get_graph_data(grid, current_pf_results, bus_idx_map)

        # Attach target and mask to data object for batch processing
        data.y_target = full_target  # [Num_Nodes, 1]
        data.mask = mask.cpu()  # [Num_Nodes]
        data.to('cpu')  # Keep on CPU until batch is formed

        batch_data_list.append(data)

        # --- Batch training: process when enough samples accumulated ---
        if len(batch_data_list) >= BATCH_SIZE:
            model.train()
            optimizer.zero_grad()

            # A. Physical batching: combine multiple small graphs into one large graph
            # This provides stable statistics for BatchNorm layers
            big_batch = Batch.from_data_list(batch_data_list).to(DEVICE)

            # B. Forward pass through GNN
            pred = model(big_batch)

            # C. Extract batched targets and masks
            target_batch = big_batch.y_target.to(DEVICE)
            mask_batch = big_batch.mask.to(DEVICE)

            # D. Compute loss only on renewable generator nodes
            if mask_batch.sum() > 0:
                loss = F.smooth_l1_loss(pred[mask_batch], target_batch[mask_batch], beta=0.1)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # E. Logging and model saving
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), SAVE_PATH)

                print(f"{epoch:<6} | {current_loss:.5f}       | Best: {best_loss:.5f}")
                log_writer.writerow([epoch, t, current_loss])

            # F. Clear batch buffer
            batch_data_list = []

    # Cleanup and evaluation
    log_f.close()
    print(f"\n🎉 Training completed! Starting evaluation...")

    # Run comprehensive evaluation on test set
    evaluate_model(model, grid, bus_idx_map, test_idx, df_load_p, df_load_q, df_sgen_p, DEVICE)


if __name__ == "__main__":
    main()