"""LINGUAL - language-guided active domain adaptation for segmentation (initial version).

Per round:
  1. the model selects uncertain patches on every target training image   (lingual/step1_selection.py)
  2. a simulated human gives language feedback per patch from the GT        (lingual/step2_feedback.py)
  3. an LLM turns the feedback into a program over the modules              (lingual/step3_program.py)
  4. the modules run and the corrected labels go into the labelled pool     (lingual/step4_modules.py)
  5. the model is fine-tuned on the pool and evaluated on the target test   (lingual/step5_train.py)

Usage:
  python main.py --cfg configs/UDIAT.txt --method RIPU
"""
import argparse
import copy
import csv
import os
import random

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lingual.annotate import label_image
from lingual.config import read_config
from dataloader.dataloader_seg import Seg_Dataset, Stack_Dataset, load_rgb
from lingual.model import seg_model
from lingual.step1_selection import select_boxes, top_up_boxes
from lingual.step5_train import evaluate, train_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--method", default=None, help="RIPU | Entropy (overrides cfg)")
    p.add_argument("--run_id", type=int, default=None, help="repeat index: seed = cfg.seed + run_id")
    return p.parse_args()


def setup(args):
    cfg = read_config(args.cfg)
    if args.method:
        cfg.method = args.method
    cfg.seed = int(getattr(cfg, "seed", 0)) + (args.run_id or 0)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.Exp_name = (f"{cfg.Exp_name}_LINGUAL_{cfg.dataset_name}_{cfg.method}_Budget_{cfg.budget}_PSize_{cfg.patch_size}"
                    f"_Epoch_{cfg.num_epochs}_Rounds_{cfg.Rounds}_Lr_{cfg.learning_rate}")
    if args.run_id is not None:
        cfg.Exp_name += f"_run{args.run_id}"

    # budget = % of image pixels over all rounds -> number of patch_size x patch_size boxes per round
    if cfg.n_budget_box == -1:
        per_round = cfg.budget / cfg.Rounds
        cfg.n_budget_box = int(np.round(per_round * 0.01 * cfg.img_size[0] * cfg.img_size[1] / cfg.patch_size ** 2))
    return cfg


def main():
    cfg = setup(parse_args())
    print(cfg)
    print("Boxes per image per round:", cfg.n_budget_box)

    target = Seg_Dataset(cfg.dataset_name, cfg.target_data_dir, "train", cfg.augmentation, AL_mode=True,
                         Exp_name=cfg.Exp_name, img_size=cfg.img_size)
    if cfg.dataset_name == "CHAOS":
        target_test = Stack_Dataset(cfg.dataset_name, cfg.target_stack_data_dir, "test", img_size=cfg.img_size)
    else:
        target_test = Seg_Dataset(cfg.dataset_name, cfg.target_data_dir, "test", img_size=cfg.img_size)
    loader = DataLoader(target, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = seg_model(num_class=cfg.num_classes).to(cfg.device)
    model.load_state_dict(torch.load(cfg.pretrained_ckpt_path, map_location=cfg.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    out_dir = os.path.join(cfg.result_save_dir, cfg.method, cfg.Exp_name)
    os.makedirs(out_dir, exist_ok=True)
    feedback_csv = open(os.path.join(out_dir, "feedback_log.csv"), "w", newline="")
    writer = csv.DictWriter(feedback_csv, fieldnames=["round", "image", "box", "label", "operation", "direction",
                                                      "instruction", "program", "error"])
    writer.writeheader()

    print("Source model on target test:")
    evaluate(model, target_test, cfg, os.path.join(out_dir, "source_model.txt"))

    for rnd in range(cfg.Rounds):
        print(f"=== Round {rnd + 1}/{cfg.Rounds} ===")

        # ---------- Steps 1-4: select patches, get feedback, correct labels ----------
        for images, gts, names, pools in loader:
            images, gts = images.to(cfg.device), gts.long()
            boxes = select_boxes(model, images, pools, cfg)  # step 1

            for i, name in enumerate(names):
                if len(boxes[i]) < cfg.n_budget_box:
                    boxes[i] = top_up_boxes(model, images[i], pools[i], boxes[i], cfg.n_budget_box - len(boxes[i]), cfg)

                model.eval()
                with torch.no_grad():
                    pred = torch.argmax(F.softmax(model(images[i:i + 1]), dim=1), dim=1)[0].cpu().numpy()

                image = load_rgb(os.path.join(target.image_dir, name), cfg.img_size)
                log = lambda row, name=name: writer.writerow({"round": rnd, "image": name, **row})
                pool = label_image(cfg, image, gts[i].numpy(), pred, boxes[i],  # steps 2-4
                                   copy.deepcopy(pools[i]).numpy(), log)
                tifffile.imwrite(os.path.join(target.AL_dir, name), pool.astype(np.uint8))
        feedback_csv.flush()

        # ---------- Step 5: train on the labelled pool ----------
        for epoch in range(cfg.num_epochs):
            loss = train_epoch(model, loader, optimizer, cfg)
            print(f"Epoch {epoch}: loss {loss:.4f}")
            if epoch % cfg.eval_interval == 0 and epoch > 0:
                evaluate(model, target_test, cfg, os.path.join(out_dir, f"round_{rnd}.txt"))

    feedback_csv.close()
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pth"))
    print("Done. Results in", out_dir)


if __name__ == "__main__":
    main()
