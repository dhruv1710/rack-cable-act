import importlib

_TASKS = {
    "cable_insertion": ("envs.tasks.cable_insertion", "CableInsertion"),
    "cable_routing": ("envs.tasks.cable_routing", "CableRouting"),
    "fuse_holder": ("envs.tasks.fuse_holder", "FuseHolder"),
    "parts_sorting": ("envs.tasks.parts_sorting", "PartsSorting"),
    "object_handoff": ("envs.tasks.object_handoff", "ObjectHandoff"),
    "bottle_binning": ("envs.tasks.bottle_binning", "BottleBinning"),
    "plate_racking": ("envs.tasks.plate_racking", "PlateRacking"),
    "pouring_beads": ("envs.tasks.pouring_beads", "PouringBeads"),
    "relative_placement": ("envs.tasks.relative_placement", "RelativePlacement"),
    "thermal_patrol": ("envs.tasks.thermal_patrol", "ThermalPatrol"),
    "drive_replacement": ("envs.tasks.drive_replacement", "DriveReplacement"),
}


def list_tasks():
    return list(_TASKS)


def make(name, **kw):
    mod, cls = _TASKS[name]
    return getattr(importlib.import_module(mod), cls)(**kw)
