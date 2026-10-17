# Source-model checkpoints

DeepLabV3 (ResNet-18 encoder) trained on the source domain. `.pth` files are git-ignored.

| Path | Source -> target | num_classes |
|---|---|---|
| `Ultrasound/deeplabv3_ultrasound.pth` | BUSI -> UDIAT | 2 |
| `MRI/deeplabv3_mri.pth` | CHAOS T1 in-phase -> out-of-phase | 5 |
| `XRAY/deeplabv3_XRAY.pth` | Montgomery -> Shenzhen | 2 |
