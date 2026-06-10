import copy
from pathlib import Path


def garment_name_from_spec(spec_file: Path):
    return spec_file.stem.split("_specification")[0]


LOWER_PANEL_PREFIXES = (
    "wb_",
    "pant_",
    "skirt_",
    "ins_skirt_",
    "skirt_panel_",
)
UPPER_TORSO_PANELS = {
    "left_btorso",
    "left_ftorso",
    "right_btorso",
    "right_ftorso",
}
UPPER_PANEL_PREFIXES = (
    "sl_",
    "left_collar_",
    "right_collar_",
)
UPPER_PANEL_NAMES = UPPER_TORSO_PANELS | {"left_hood", "right_hood"}
UPPER_HEM_LABEL = "upper_hem"
WAISTBAND_LABEL = "lower_interface"


def classify_panel(name, panel):
    """Classify a combined pattern panel as Upper or Lower."""
    if name.startswith(LOWER_PANEL_PREFIXES):
        return "Lower"
    if (name in UPPER_PANEL_NAMES or
            name.startswith(UPPER_PANEL_PREFIXES) or
            "_sleeve_" in name):
        return "Upper"

    label = panel.get("label")
    if label == "leg":
        return "Lower"
    if label == "arm":
        return "Upper"

    raise ValueError(
        f"Unable to classify panel '{name}'. Add it to the Upper/Lower split rules."
    )


def stitch_panel_names(stitch):
    return [item["panel"] for item in stitch if isinstance(item, dict) and "panel" in item]


def has_cross_group_stitches(stitches, panel_groups):
    for stitch in stitches:
        groups = {panel_groups[name] for name in stitch_panel_names(stitch)}
        if len(groups) > 1:
            return True
    return False


def label_upper_hem_edges(spec):
    """Add upper_hem labels to the lowest torso edge in each torso panel."""
    panels = spec["pattern"]["panels"]
    already_labeled = any(
        edge.get("label") == UPPER_HEM_LABEL
        for panel in panels.values()
        for edge in panel.get("edges", [])
    )
    if already_labeled:
        return

    for panel_name in UPPER_TORSO_PANELS:
        panel = panels.get(panel_name)
        if panel is None:
            continue

        vertices = panel.get("vertices", [])
        edges = panel.get("edges", [])
        if not vertices or not edges:
            continue

        translation_y = panel.get("translation", [0.0, 0.0, 0.0])[1]
        best_edge = None
        best_y = None
        for edge in edges:
            endpoints = edge.get("endpoints", [])
            if len(endpoints) != 2 or edge.get("label"):
                continue
            y0 = vertices[endpoints[0]][1] + translation_y
            y1 = vertices[endpoints[1]][1] + translation_y
            avg_y = 0.5 * (y0 + y1)
            if best_y is None or avg_y < best_y:
                best_y = avg_y
                best_edge = edge

        if best_edge is not None:
            best_edge["label"] = UPPER_HEM_LABEL


def validate_lower_waistband_label(spec):
    panels = spec["pattern"]["panels"]
    for panel_name in ("wb_back", "wb_front"):
        panel = panels.get(panel_name)
        if panel is None:
            continue
        if any(edge.get("label") == WAISTBAND_LABEL for edge in panel.get("edges", [])):
            return
    raise ValueError(
        f"Lower garment needs a '{WAISTBAND_LABEL}' edge on wb_back/wb_front "
        "for tucked/untucked constraints."
    )


def filtered_spec(source_spec, panel_groups, target_group):
    spec = copy.deepcopy(source_spec)
    pattern = spec["pattern"]
    panels = {
        name: panel
        for name, panel in pattern["panels"].items()
        if panel_groups[name] == target_group
    }
    stitches = [
        stitch
        for stitch in pattern.get("stitches", [])
        if all(name in panels for name in stitch_panel_names(stitch))
    ]
    pattern["panels"] = panels
    pattern["stitches"] = stitches
    pattern["panel_order"] = [
        name for name in pattern.get("panel_order", panels.keys()) if name in panels
    ]
    return spec
