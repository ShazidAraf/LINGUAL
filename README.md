# LINGUAL: Language-Guided Active Domain Adaptation for Medical Image Segmentation

A segmentation model trained on a source domain is adapted to a target domain with only a small
annotation budget. The annotator does not draw masks. They give short **language feedback**
on the patches where the model is most uncertain, and the feedback is turned into mask edits automatically.

## Pipeline

Each active-learning round runs five steps:

| Step | What happens | Code |
|---|---|---|
| 1. Select | The model scores every pixel's uncertainty (RIPU or Entropy) and picks the top 21x21 patches per image | `lingual/step1_selection.py`, `Acquisition_Function/` |
| 2. Feedback | A simulated annotator compares prediction and ground truth inside each patch and writes feedback such as *"Expand the border to top-left, Remove the fragments, Smooth the Border"* | `lingual/step2_feedback.py` |
| 3. Program | An LLM (GPT-3.5, VisProg-style few-shot prompt) converts the feedback into a program over the editing modules | `lingual/step3_program.py` |
| 4. Execute | Modules edit the mask inside the patch: `EXPAND` / `SHRINK` (GraCo granularity-controllable interactive segmentation), `FILL`, `REMOVE`, `SMOOTH`. The edited patch goes into the labelled pool | `lingual/step4_modules.py`, `lingual/graco_executor.py`, `lingual/annotate.py` |
| 5. Train | The model is fine-tuned on labelled-pool pixels only (unlabelled pixels = 255, ignored) and evaluated on the target test set | `lingual/step5_train.py` |

Ground truth is used only by the simulated annotator (step 2) and for evaluation.
Whole-patch answers (`Entire region Foreground`, `Entire region Background`, `No change`) skip steps 3 and 4.

## Repository layout

```
main.py                     round loop (steps 1-5)
run.sh                      SLURM launcher
configs/                    UDIAT.txt, CHAOS.txt, Shenzhen.txt (local/ = your own paths, git-ignored)
lingual/                    LINGUAL code (one file per step + data, model, metrics)
Acquisition_Function/       RIPU and Entropy acquisition + losses
third_party/graco/isegm/    GraCo inference code (Apache-2.0, see third_party/graco/LICENSE)
environment.yml, requirements.txt
```

## Installation

```bash
conda env create -f environment.yml
conda activate lingual
```
Tested with Python 3.10, PyTorch 2.0.1, CUDA 11.8. A GPU is required (GraCo runs on `cuda:0`).

## Weights (not included)

1. **GraCo**: download `sbd_vit_large.pth` (SimpleClick) and `GraCo_large_lora.pth` from the
   [GraCo repository](https://github.com/Zhao-Yian/GraCo) and arrange them as
   ```
   <weights>/simple_click/sbd_vit_large.pth
   <weights>/graco/GraCo_large_lora.pth
   ```
   Put them in `weights/` inside this repo (default), or `export GRACO_WEIGHTS=<weights>`.
2. **Source segmentation model**: a DeepLabV3 (ResNet-18) checkpoint trained on the source domain.
   Default location: `checkpoints/<name>.pth` (see `pretrained_ckpt_path` in the config).

## Data

Each dataset folder must look like:
```
<target_data_dir>/train/data/*.png    <target_data_dir>/train/labels/*.png
<target_data_dir>/test/data/*.png     <target_data_dir>/test/labels/*.png
```
Binary labels are stored as 0/255. CHAOS labels are stored in steps of 63, and its test set is a set of 3D tif stacks (`target_stack_data_dir`).

Adaptation settings used: BUSI → UDIAT (ultrasound), CHAOS T1 in-phase → out-of-phase (MRI), Montgomery → Shenzhen (X-ray).

## Run

The configs expect `data/`, `checkpoints/` and `weights/` inside the repo (all git-ignored; symlinks are fine).
Otherwise edit the paths in `configs/<dataset>.txt`. Machine-specific copies can go in `configs/local/` (git-ignored). Then:
```bash
export OPENAI_API_KEY=...          # needed for program_mode = llm
python main.py --cfg configs/UDIAT.txt --method RIPU
# or on SLURM, from this folder:
sbatch run.sh configs/UDIAT.txt RIPU
```
`--method` is `RIPU` or `Entropy`. `--run_id K` repeats a run with seed `seed + K`.

### Main config keys

| Key | Meaning |
|---|---|
| `budget` | % of image pixels labelled, summed over all rounds |
| `Rounds` | number of active-learning rounds |
| `patch_size` | side of a selected patch (odd) |
| `program_mode` | `llm` (paper setting) or `template` (rule-based program, offline, no API key) |
| `bool_region_mask` | `False` reproduces the original behaviour. When a direction is given, FILL/REMOVE/SMOOTH index the image with a 0/1 integer array, so they barely change it. `True` uses a boolean mask, which is the intended behaviour but a method change |
| `num_epochs`, `learning_rate`, `batch_size` | fine-tuning per round |

## Outputs

`results/<method>/<Exp_name>/`
- `source_model.txt`: target-test metrics before adaptation
- `round_<r>.txt`: DSC, RAVD, HD95, ASSD per class during each round
- `feedback_log.csv`: one row per patch with the operation, direction, instruction, generated program and any executor error
- `final_model.pth`

Labelled-pool masks are written to `<target_data_dir>/train/AL_mask/<Exp_name>/`.

## Acknowledgements

- RIPU: Xie et al., *Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation*, CVPR 2022.
- GraCo: Zhao et al., *GraCo: Granularity-Controllable Interactive Segmentation*, CVPR 2024 (code in `third_party/graco`, Apache-2.0), built on SimpleClick.
- VisProg: Gupta & Kembhavi, *Visual Programming: Compositional Visual Reasoning without Training*, CVPR 2023 (program-generation prompt style).
