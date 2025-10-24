def get_bus_by_name(grid_gc, bus_name: str | int):
    key = str(bus_name)
    for b in grid_gc.buses:
        if b.name == key:
            return b
    raise ValueError(f"Bus {bus_name} not found")

def get_generators_at_bus(grid_gc, bus_name: str | int):
    b = get_bus_by_name(grid_gc, bus_name)
    return [g for g in grid_gc.generators if g.bus is b]