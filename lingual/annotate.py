"""Steps 2-4 for one image: for every selected box, simulate feedback, build a program, run it,
and write the corrected labels into the labelled pool mask (only inside the box).
Same logic as LINGUAL_v3.Apply_LINGUAL with refine_boxes = False and use_only_improved_mask = False.
"""
import traceback

from lingual.step2_feedback import simulate_feedback
from lingual.step3_program import make_program
from lingual.step4_modules import execute


def label_image(cfg, image, ref_total, pred_total, boxes, pool_mask, log):
    """
    image      : RGB PIL at cfg.img_size (input to the executors)
    ref_total  : HxW GT label map (used only by the simulated annotator)
    pred_total : HxW current model prediction (label map); updated box by box
    boxes      : list of [x1, y1, x2, y2]
    pool_mask  : HxW labelled-pool mask (255 = unlabelled); returned with the boxes filled in
    log        : callable(dict) that records one row per box
    """
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        operation, direction, keywords, ref, pred, label = simulate_feedback(ref_total, pred_total, [x1, y1, x2, y2], cfg)
        row = dict(box=[x1, y1, x2, y2], label=int(label), operation=operation, direction=direction,
                   instruction=", ".join(keywords), program="", error="")

        if direction == -1:  # whole-box answers: no program needed
            if operation == "No change":
                pool_mask[y1:y2, x1:x2] = pred_total[y1:y2, x1:x2]
            elif operation == "Foreground":
                pool_mask[y1:y2, x1:x2] = label
                pred_total[y1:y2, x1:x2] = label
            elif operation == "Background":
                pool_mask[y1:y2, x1:x2] = 0
                pred_total[y1:y2, x1:x2] = 0
            # "Ignore": box stays unlabelled
            log(row)
            continue

        instruction, program = make_program(keywords, cfg)
        row.update(instruction=instruction, program=program.replace("\n", " | "))

        state = dict(IMAGE=image, MASK=pred, ROI=[y1, x1, y2 - y1, x2 - x1], instruction=instruction,
                     fragment_threshold=cfg.fragment_threshold,
                     bool_region_mask=getattr(cfg, "bool_region_mask", False))
        try:
            result = execute(program, state)
        except Exception as e:  # executor failure -> keep the model prediction in this box
            row["error"] = f"{type(e).__name__}: {e}"
            if getattr(cfg, "DEBUG_MODE", False):
                traceback.print_exc()
            result = pred

        pred_total[y1:y2, x1:x2] = label * result[y1:y2, x1:x2]
        pool_mask[y1:y2, x1:x2] = pred_total[y1:y2, x1:x2]
        log(row)
    return pool_mask
