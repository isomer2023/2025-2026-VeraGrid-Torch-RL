import os, glob
import numpy as np
import torch

DATA_DIR = "dataset_output_1mv_urban_dynamic_topo_FULL_STATE"
STATS_PATH = os.path.join(DATA_DIR, "stats.pt")

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt")))
    assert files, f"No chunk files in {DATA_DIR}"
    print("Found chunks:", len(files), "first:", files[0])

    # ---- load one chunk & one sample ----
    chunk = torch.load(files[0], weights_only=False, map_location="cpu")
    assert isinstance(chunk, list) and len(chunk) > 0
    s = chunk[0]

    # ---- keys check ----
    must_keys = ["x","y","mask","adj","attn_bias","edge_index","edge_attr"]
    print("Sample keys:", list(s.keys()))
    for k in must_keys:
        assert k in s, f"Missing key: {k}"
    print("✅ All required keys exist.")

    x = s["x"]          # (C,N)
    y = s["y"]          # (4,N)
    mask = s["mask"]    # (N,)
    adj = s["adj"]      # (N,N)
    bias = s["attn_bias"]   # (N,N)
    ei = s["edge_index"]    # (2,E)
    ea = s["edge_attr"]     # (E,2)

    print("\n--- shapes ---")
    print("x:", tuple(x.shape), x.dtype)
    print("y:", tuple(y.shape), y.dtype)
    print("mask:", tuple(mask.shape), mask.dtype, "mask_sum:", int(mask.sum()))
    print("adj:", tuple(adj.shape), adj.dtype)
    print("attn_bias:", tuple(bias.shape), bias.dtype)
    print("edge_index:", tuple(ei.shape), ei.dtype)
    print("edge_attr:", tuple(ea.shape), ea.dtype)

    # ---- expected dims ----
    C, N = x.shape
    assert y.shape[1] == N, "y N mismatch"
    assert adj.shape == (N, N), "adj shape mismatch"
    assert bias.shape == (N, N), "bias shape mismatch"
    assert ei.shape[0] == 2, "edge_index first dim must be 2"
    assert ea.shape[1] == 2, "edge_attr must be (E,2)"

    # ---- sanity: vn_kv channel non-trivial ----
    # assumes vn_kv is x[3]
    if C >= 4:
        vn = x[3].numpy()
        print("\n--- vn_kv stats ---")
        print("vn_kv min/max:", float(vn.min()), float(vn.max()), "std:", float(vn.std()))
    else:
        print("\n⚠️ x has C<4, vn_kv not present? C =", C)

    # ---- sanity: attn_bias not all zeros ----
    b = bias.numpy()
    print("\n--- attn_bias stats ---")
    print("bias min/max:", float(b.min()), float(b.max()), "mean:", float(b.mean()))
    print("bias diag max abs:", float(np.max(np.abs(np.diag(b)))))

    # ---- sanity: edge_attr not all zeros ----
    ea_np = ea.numpy()
    print("\n--- edge_attr stats ---")
    print("r_pu min/max:", float(ea_np[:,0].min()), float(ea_np[:,0].max()), "mean:", float(ea_np[:,0].mean()))
    print("x_pu min/max:", float(ea_np[:,1].min()), float(ea_np[:,1].max()), "mean:", float(ea_np[:,1].mean()))
    zero_frac = float(np.mean(np.all(np.isclose(ea_np, 0.0), axis=1)))
    print("edge_attr all-zero row fraction:", zero_frac)

    # ---- y sanity: alpha in [0,1], vm around 1, sin/cos bounded ----
    y_np = y.numpy()
    alpha = y_np[0]
    vm = y_np[1]
    s1 = y_np[2]
    c1 = y_np[3]
    print("\n--- y stats ---")
    print("alpha min/max:", float(alpha.min()), float(alpha.max()))
    print("vm min/max:", float(vm.min()), float(vm.max()), "mean:", float(vm.mean()))
    print("sin min/max:", float(s1.min()), float(s1.max()), "mean:", float(s1.mean()))
    print("cos min/max:", float(c1.min()), float(c1.max()), "mean:", float(c1.mean()))
    # check sin^2+cos^2 ~ 1
    sc = s1*s1 + c1*c1
    print("sin^2+cos^2 mean:", float(sc.mean()), "min:", float(sc.min()), "max:", float(sc.max()))

    # ---- stats.pt check ----
    assert os.path.exists(STATS_PATH), "stats.pt missing"
    st = torch.load(STATS_PATH, weights_only=False, map_location="cpu")
    xm, xs = st["x_mean"], st["x_std"]
    print("\n--- stats.pt ---")
    print("x_mean:", xm, "shape:", tuple(xm.shape))
    print("x_std :", xs, "shape:", tuple(xs.shape))
    print("✅ stats.pt exists and readable.")

    # ---- check edge consistency across a few samples ----
    print("\n--- edge consistency check (first 5 samples) ---")
    for i in range(min(5, len(chunk))):
        eii = chunk[i]["edge_index"]
        eaa = chunk[i]["edge_attr"]
        assert eii.shape == ei.shape and eaa.shape == ea.shape, "Edge shape differs across samples!"
    print("✅ edge_index/edge_attr shapes consistent in this chunk.")

    print("\n🎉 Dataset looks readable and non-trivial.")

if __name__ == "__main__":
    main()
