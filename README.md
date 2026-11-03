# LINGUAL: Language-Guided Active Domain Adaptation for Medical Image Segmentation

LINGUAL adapts a segmentation model trained on a source domain to a new target domain with a small
annotation budget. Instead of drawing masks, the annotator gives short **language feedback** on the
image patches where the model is most uncertain. The feedback is converted into a program that edits
the predicted mask, and the edited patches are used to fine-tune the model.

## Method

Adaptation runs for several rounds. Each round has five steps.

1. **Uncertain patch selection.** The model predicts every target training image. An acquisition
   function (RIPU or Entropy) scores each pixel, and the top-scoring square patches are selected.
   Patches that are already labelled are never selected again.
2. **Language feedback.** For every selected patch, the annotator describes what is wrong with the
   prediction, for example *"Expand the border to top-left, Remove the fragments, Smooth the Border"*.
   Whole-patch answers are *"Entire region Foreground"*, *"Entire region Background"* and *"No change"*.
   The annotator is simulated from the ground truth.
3. **Program generation.** A large language model turns the feedback into a short program over the
   editing modules, following a few-shot visual-programming prompt.
4. **Program execution.** The modules edit the mask inside the patch:
   - `DIRECTION` reads the direction named in the feedback
   - `EXPAND` / `SHRINK` move the boundary using GraCo, a granularity-controllable interactive segmentation model
   - `FILL` closes holes, `REMOVE` deletes small fragments, `SMOOTH` smooths the border

   The edited patch is written into the labelled pool. Pixels outside selected patches stay unlabelled.
5. **Training.** The model is fine-tuned on the labelled-pool pixels only and evaluated on the target test set.

## Repository structure

```
main.py                   runs the rounds (steps 1-5)
run.sh                    SLURM launcher
configs/                  one config per dataset
lingual/                  step1_selection.py ... step5_train.py, modules, GraCo executor, model, metrics
Acquisition_Function/     RIPU and Entropy acquisition functions and losses
dataloader/               2D datasets and 3D stacks, labelled-pool masks
checkpoint/               source-domain segmentation models
third_party/graco/        GraCo inference code (Apache-2.0)
```

## Installation

```bash
git clone <this-repo>
cd LINGUAL
conda env create -f environment.yml
conda activate lingual
```

Requirements: Python 3.10, PyTorch 2.0.1, CUDA 11.8 and one GPU.

## Weights

**GraCo.** Download `sbd_vit_large.pth` and `GraCo_large_lora.pth` from the
[GraCo repository](https://github.com/Zhao-Yian/GraCo) and place them as:
```
weights/simple_click/sbd_vit_large.pth
weights/graco/GraCo_large_lora.pth
```

**Source model.** Put the source-domain DeepLabV3 checkpoint in `checkpoint/<modality>/`, e.g. `checkpoint/Ultrasound/deeplabv3_ultrasound.pth`.
.

## Data

Each dataset folder follows this layout:
```
<dataset>/train/data/     <dataset>/train/labels/
<dataset>/test/data/      <dataset>/test/labels/
```
Binary labels use 0/255. CHAOS labels use multiples of 63, and the CHAOS test set is given as 3D tif stacks.

Provided configs: BUSI to UDIAT (ultrasound), CHAOS T1 in-phase to out-of-phase (MRI),
Montgomery to Shenzhen (chest X-ray).

## Usage

1. Set the paths in `configs/<dataset>.txt`.
2. Set your OpenAI key (used for program generation):
   ```bash
   export OPENAI_API_KEY=<your-key>
   ```
3. Run:
   ```bash
   python main.py --cfg configs/UDIAT.txt --method RIPU
   ```
   or on a SLURM cluster:
   ```bash
   sbatch run.sh configs/UDIAT.txt RIPU
   ```

Arguments:
- `--cfg` config file
- `--method` acquisition function, `RIPU` or `Entropy`
- `--run_id K` repeat a run with seed `seed + K`

## Configuration

| Key | Description |
|---|---|
| `target_data_dir` | target dataset folder |
| `target_stack_data_dir` | 3D test stacks (CHAOS only) |
| `pretrained_ckpt_path` | source-domain segmentation model |
| `graco_weights_dir` | folder containing the GraCo weights |
| `result_save_dir` | output folder |
| `pool_dir` | folder for the labelled-pool masks |
| `budget` | percentage of image pixels annotated over all rounds |
| `Rounds` | number of rounds |
| `patch_size` | side length of a selected patch (odd) |
| `program_mode` | `llm` to generate programs with the LLM, `template` to build them from fixed rules |
| `bool_region_mask` | region indexing used by directional FILL / REMOVE / SMOOTH (default `False`) |
| `fragment_threshold` | minimum component size kept by `REMOVE` |
| `num_epochs`, `batch_size`, `learning_rate` | fine-tuning per round |
| `num_classes`, `img_size`, `seed` | dataset and run settings |

## Outputs

Written to `<result_save_dir>/<method>/<Exp_name>/`:
- `source_model.txt` target test metrics before adaptation
- `round_<r>.txt` DSC, RAVD, HD95 and ASSD per class during round `r`
- `feedback_log.csv` feedback, generated program and result for every patch
- `final_model.pth` adapted model

## Acknowledgements

- RIPU: Xie et al., *Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation*, CVPR 2022.
- GraCo: Zhao et al., *GraCo: Granularity-Controllable Interactive Segmentation*, CVPR 2024.
- VisProg: Gupta and Kembhavi, *Visual Programming: Compositional Visual Reasoning without Training*, CVPR 2023.
