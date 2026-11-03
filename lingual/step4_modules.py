"""STEP 4 - execute the program with the LINGUAL modules.

Program syntax (one step per line):  OUT=MODULE(arg=value, ...)
Modules
  DIRECTION(command)  : directions named next to `command` in the instruction text
  EXPAND / SHRINK     : GraCo boundary executor (graco_executor.MODIFY)
  FILL                : morphological closing (fills holes)
  REMOVE              : drops connected components < cfg.fragment_threshold px
  SMOOTH              : Gaussian blur + threshold
  RESULT(var)         : returns var
Every module only writes inside the box (and inside the direction half-plane if a direction is given).
State keys: IMAGE (PIL), MASK (HxW binary), ROI ([top_y, top_x, h, w]), instruction, fragment_threshold.
"""
import copy
import io
import re
import tokenize

import cv2
import numpy as np

from lingual.graco_executor import MODIFY
from lingual.regions import weight_highlight_region

ALL_DIRECTIONS = sorted(["top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"],
                        key=len, reverse=True)


def parse_step(step):
    """'OUT=NAME(a=x, b=y)' -> (OUT, NAME, {a: x, b: y})."""
    tokens = list(tokenize.generate_tokens(io.StringIO(step).readline))
    arg_tokens = [t.string for t in tokens[4:-3] if t.string not in (",", "=")]
    args = {arg_tokens[2 * i]: arg_tokens[2 * i + 1].strip("'\"") for i in range(len(arg_tokens) // 2)}
    return tokens[0].string, tokens[2].string, args


def directions_for(keyword, instruction):
    """Directions named in every sentence that contains all words of `keyword`."""
    words = re.findall(r"[a-z]+", keyword.lower())
    found = set()
    for sentence in re.split(r"[.?!]", instruction):
        if all(w in sentence.lower() for w in words):
            found.update(re.findall(r"\b(" + "|".join(map(re.escape, ALL_DIRECTIONS)) + r")\b", sentence.lower()))
    return list(found)


def region_of(mask, box, direction):
    """Where a module may write: the box, intersected with the direction half-plane if given."""
    ty, tx, h, w = box
    region = np.zeros_like(mask, dtype=np.uint8)
    region[ty:ty + h, tx:tx + w] = 1
    if len(direction):
        center = np.array([ty + h // 2, tx + w // 2])
        region = np.logical_and(region, weight_highlight_region(mask, center, direction, adjust_len=0)).astype(np.uint8)
    return region


def _paste(mask, edited, box, direction, bool_region):
    out = copy.deepcopy(mask)
    if len(direction) == 0:
        ty, tx, h, w = box
        out[ty:ty + h, tx:tx + w] = edited[ty:ty + h, tx:tx + w]
    else:
        sel = region_of(mask, box, direction)
        # bool_region_mask = False indexes with the 0/1 array, True with a boolean mask
        if bool_region:
            sel = sel.astype(bool)
        out[sel] = edited[sel]
    return out


def smooth(mask, kernel_size=11, sigma=5, threshold=0.5):
    blurred = cv2.GaussianBlur((mask * 255).astype(np.uint8), (kernel_size, kernel_size), sigma) / 255.0
    return (blurred > threshold).astype(np.uint8)


def remove_small(mask, threshold):
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [l for l in range(1, n) if stats[l, cv2.CC_STAT_AREA] >= threshold]
    return np.isin(labels, keep).astype(np.uint8)


def fill_holes(mask, kernel_size=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return (cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0).astype(np.uint8)


def execute(program, state):
    """Run the program line by line on `state`; returns the RESULT value."""
    out = None
    for line in program.split("\n"):
        if not line.strip():
            continue
        var, name, args = parse_step(line)
        if name == "DIRECTION":
            out = directions_for(args["command"], state["instruction"])
        elif name == "RESULT":
            return state[args["var"]]
        else:
            mask, box = state[args["mask"]], state["ROI"]
            direction = state.get(args.get("direction", ""), "")
            bool_region = state.get("bool_region_mask", False)
            if name in ("EXPAND", "SHRINK"):
                out = MODIFY(np.array(state["IMAGE"]), copy.deepcopy(mask), box=box, direction=direction,
                             mode=name.lower())
            elif name == "FILL":
                out = _paste(mask, fill_holes(mask), box, direction, bool_region)
            elif name == "REMOVE":
                out = _paste(mask, remove_small(mask, state["fragment_threshold"]), box, direction, bool_region)
            elif name == "SMOOTH":
                out = _paste(mask, smooth(mask), box, direction, bool_region)
            else:
                raise ValueError(f"Unknown module {name}")
        state[var] = out
    return out
