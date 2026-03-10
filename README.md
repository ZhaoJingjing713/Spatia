<div align="center">

<h1 align="center">
  Spatia: Video Generation with Updatable Spatial Memory
  <br>
  <sub>CVPR 2026</sub>
</h1>

<p align="center">
  <strong>Long-horizon, spatially consistent video generation enabled by persistent 3D scene point clouds and dynamic-static disentanglement.</strong>
</p>

<div align="center">
  <a href="https://github.com/ZhaoJingjing713">Jinjing Zhao</a><sup>*1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=-ncz2s8AAAAJ">Fangyun Wei</a><sup>*2</sup>&nbsp;&nbsp;
  <a href="https://www.liuzhening.top">Zhening Liu</a><sup>3</sup>&nbsp;&nbsp;
  <a href="https://hongyanz.github.io/">Hongyang Zhang</a><sup>4</sup>&nbsp;&nbsp;
  <a href="http://changxu.xyz/">Chang Xu</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=djk5l-4AAAAJ">Yan Lu</a><sup>2</sup>
</div>

<p align="center" style="font-size: 0.9em; color: #666;">
  <sup>1</sup>The University of Sydney&nbsp;&nbsp;
  <sup>2</sup>Microsoft Research&nbsp;&nbsp;
  <sup>3</sup>HKUST&nbsp;&nbsp;
  <sup>4</sup>University of Waterloo
  <br>
  <small><sup>*</sup>Equal Contribution</small>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2512.15716">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat&labelColor=555555&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  &nbsp;
  <a href="https://zhaojingjing713.github.io/Spatia/">
    <img src="https://img.shields.io/badge/Project-Page-4F46E5?style=flat&labelColor=555555&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  &nbsp;
  <a href="https://huggingface.co/Jinjing713/Spatia">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-FFB300?style=flat&labelColor=555555&logo=huggingface&logoColor=white" alt="Hugging Face">
  </a>
</p>

</div>

---

## Clone This Repo

This repository has multiple branches. The `page` branch contains the project website and many large asset files, so for inference usage you should clone only the `main` branch:

```bash
git clone --single-branch --branch main https://github.com/ZhaoJingjing713/Spatia.git
cd Spatia
```

If you have already cloned the full repository and want to avoid fetching other branches later, keep working on `main` and avoid checking out `page`.

---

## Overview

This release provides the inference pipeline for **Spatia**, built on top of:

- **Wan2.2-TI2V-5B** as the base video generator
- a **control / VACE checkpoint** for spatial guidance
- a **LoRA checkpoint** for autoregressive long-horizon generation

Main entry point:

```bash
python inference.py
```

At a high level, the pipeline:

1. Reconstructs and updates a 3D scene point cloud with MapAnything.
2. Renders point-guided control videos from target camera trajectories.
3. Generates spatially consistent clips with Wan2.2.
4. Reuses history frames and matched reference views for long-horizon rollout.

---

## Repository Layout

```text
.
├── inference.py
├── install.sh
├── download.sh
├── requirements.txt
├── test_cases/
├── utils/
└── wan/
```

Key files:

- `inference.py`: end-to-end inference entry
- `install.sh`: environment setup script
- `download.sh`: base Wan2.2 checkpoint download helper
- `test_cases/`: example prompts, intrinsics, and camera trajectories

---

## Installation

### 1. Create the environment

First create and activate a Conda environment with Python 3.12:

```bash
conda create -n spatia python=3.12 -y
conda activate spatia
```

Then run the install script. It supports both `cuda` and `rocm`.

```bash
bash install.sh cuda
```

or

```bash
bash install.sh rocm
```

You can also specify a custom build directory:

```bash
bash install.sh cuda ./env
```

The script installs:

- PyTorch
- FlashAttention
- DiffSynth-Studio
- PyTorch3D
- MapAnything
- Python dependencies from `requirements.txt`

### 2. Download the base Wan2.2 model

Download **Wan2.2-TI2V-5B** to `model_weights/Wan2.2-TI2V-5B`:

```bash
bash download.sh
```

Equivalent manual command:

```bash
pip install -U "huggingface_hub[cli]<1.0.0"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./model_weights/Wan2.2-TI2V-5B
```

### 3. Download Spatia weights

Download the Spatia checkpoints from Hugging Face:

```bash
hf download Jinjing713/Spatia --local-dir ./checkpoints/Spatia
```

You need:

- one **control / VACE checkpoint**
- one **LoRA checkpoint**

Then point `--vace_path` and `--lora_path` to the downloaded files.

Example:

```bash
python inference.py \
  --vace_path ./checkpoints/Spatia/step-8500.safetensors \
  --lora_path ./checkpoints/Spatia/lora_weights_10000.safetensors
```

If the filenames in your local download differ, just update the paths accordingly.

---

## Input Format

Each run requires:

- one **starting image** via `--img_path`
- one **camera trajectory file** via `--camera_w2c_path`
- one **intrinsics file** via `--camera_intrinsics_path`
- one **prompt** or **prompt file**

### `w2c.txt`

`w2c.txt` stores one camera extrinsic per line in JSON-style list format.

- Coordinate convention: **OpenCV camera coordinates**
- Matrix type: **world-to-camera**
- Supported shapes per line:
  - `3x4`
  - homogeneous `4x4`

Examples:

```text
[[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3]]
```

```text
[[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3], [0, 0, 0, 1]]
```

### `intrinsics.txt`

`intrinsics.txt` stores **normalized intrinsics**:

```text
[fx fy cx cy]
```

Here:

- `fx`, `cx` are normalized by image width
- `fy`, `cy` are normalized by image height

Pixel-space intrinsics are reconstructed internally using runtime `--width` and `--height`.

### `--mask_path`

`--mask_path` specifies **foreground object masks** for dynamic/static disentanglement and controllable generation.

The foreground mask marks dynamic or editable foreground regions in the input image or video. It is used during spatial memory construction so the model can better separate:

- **static scene structure**
- **foreground dynamic content**

Supported formats:

- one or more **binary mask images**
- one `.npy` file with shape `[N, H, W]`

Typical usage:

- a single foreground mask for the first frame
- frame-aligned masks for a full sequence

Notes:

- Image masks should be binary foreground masks.
- For `.npy`, each slice should be one binary mask frame.
- If the masks are frame-aligned, use `--per_frame_mask`.

### `prompt.txt`

`prompt.txt` can contain:

- a single prompt line, reused for all rounds
- multiple prompt lines, one prompt per round

---

## Quick Start

Minimal example:

```bash
python inference.py \
  --img_path path/to/input.jpg \
  --camera_w2c_path test_cases/case_2/w2c.txt \
  --camera_intrinsics_path test_cases/case_2/intrinsics.txt \
  --prompt_path test_cases/case_2/prompt.txt \
  --save_path test_cases/case_2/output.mp4 \
  --vace_path ./checkpoints/Spatia/step-8500.safetensors \
  --lora_path ./checkpoints/Spatia/lora_weights_10000.safetensors
```

---

## CLI Arguments

### Core paths

| Argument | Default | Meaning |
| --- | --- | --- |
| `--img_path` | `test_cases/case_2/img.jpg` | Starting image for the first frame. |
| `--camera_w2c_path` | `test_cases/case_2/w2c.txt` | Camera trajectory in OpenCV world-to-camera format. |
| `--camera_intrinsics_path` | `test_cases/case_2/intrinsics.txt` | Normalized camera intrinsics file. |
| `--save_path` | `test_cases/case_2/output.mp4` | Final output video path. |
| `--work_dir` | `test_cases/case_2/output/` | Directory for intermediate reconstruction and rendering assets. |
| `--vace_path` | project-specific default | Control / VACE checkpoint path. |
| `--control_path` | `""` | Alias of `--vace_path`. |
| `--lora_path` | project-specific default | LoRA checkpoint path. |

### Prompt and masks

| Argument | Default | Meaning |
| --- | --- | --- |
| `--prompt` | `""` | Inline text prompt. Used if `--prompt_path` is empty. |
| `--prompt_path` | `test_cases/case_2/prompt.txt` | Prompt file path. |
| `--mask_path` | `[]` | Foreground mask input for dynamic/static disentanglement. Supports binary image masks or one `.npy` file of shape `[N, H, W]`. |
| `--per_frame_mask` | `True` | Treat provided masks as frame-aligned masks instead of one shared mask. |

### Resolution and rollout

| Argument | Default | Meaning |
| --- | --- | --- |
| `--width` | `1248` | Output video width. |
| `--height` | `704` | Output video height. |
| `--max_frames` | `194` | Maximum number of camera poses to use. |
| `--first_round_frames` | `121` | Number of frames generated in the first round. |
| `--round_frames` | `81` | Number of frames generated in later rounds. |
| `--hist_frames` | `9` | Number of history frames reused across rounds. |
| `--fps` | `24` | Output video FPS. |

### Reference-frame selection

| Argument | Default | Meaning |
| --- | --- | --- |
| `--map_fps` | `4` | Sampling FPS used for MapAnything reconstruction. |
| `--ref_fps` | `2` | Sampling FPS used for target reference matching. |
| `--ref_hist_fps` | `6` | Sampling FPS used for history-frame reference matching. |
| `--ref_num` | `7` | Number of matched reference frames used per round. |

### Diffusion settings

| Argument | Default | Meaning |
| --- | --- | --- |
| `--num_inference_steps` | `40` | Number of denoising steps per generated clip. |
| `--cfg_scale` | `3.5` | Classifier-free guidance scale. |
| `--sigma_shift` | `5.0` | Scheduler sigma shift. |
| `--sampler` | `uni_pc` | Sampler type. |
| `--vace_scale` | `1.0` | Control branch strength. |
| `--seed` | `20917` | Random seed. |

### Subprocess and reconstruction

| Argument | Default | Meaning |
| --- | --- | --- |
| `--map_python` | `sys.executable` | Python executable used for MapAnything subprocesses. |
| `--render_python` | `sys.executable` | Python executable used for point rendering subprocesses. |
| `--map_device` | `cuda` | Device for MapAnything. |
| `--render_device` | `cuda` | Device for point rendering. |
| `--map_conf_percentile` | `0.0` | Confidence filtering percentile for reconstructed points. |
| `--map_voxel_size` | `0.01` | Voxel size for MapAnything downsampling. |
| `--render_voxel_size` | `0.01` | Initial voxel size for point rendering. |
| `--render_voxel_size_step` | `0.005` | Voxel size increment per round. |
| `--render_batchsize` | `64` | Batch size for point rendering. |
| `--point_retrieval_batch_size` | `10000000` | Batch size for frustum-based point matching. |
| `--force_rebuild_intermediate` | `False` | Recompute intermediate outputs even if cached results exist. |
| `--verbose_subprocess` | `False` | Print subprocess logs instead of running quietly. |

---

## Test Cases

The repository includes three example camera/prompt setups:

- `test_cases/case_1`
- `test_cases/case_2`
- `test_cases/case_3`

Each case includes:

- `prompt.txt`
- `w2c.txt`
- `intrinsics.txt`

`case_3` additionally uses:

- `mask.png`

The starting image is **not included**. Before running the examples below, place an input image in each case directory, for example:

```text
test_cases/case_1/img.jpg
test_cases/case_2/img.jpg
test_cases/case_3/img.jpg
```

### Case 1

```bash
python inference.py \
  --img_path test_cases/case_1/img.jpg \
  --camera_w2c_path test_cases/case_1/w2c.txt \
  --camera_intrinsics_path test_cases/case_1/intrinsics.txt \
  --prompt_path test_cases/case_1/prompt.txt \
  --save_path test_cases/case_1/output.mp4 \
  --work_dir test_cases/case_1/output \
  --vace_path ./checkpoints/Spatia/step-8500.safetensors \
  --lora_path ./checkpoints/Spatia/lora_weights_10000.safetensors
```

### Case 2

```bash
python inference.py \
  --img_path test_cases/case_2/img.jpg \
  --camera_w2c_path test_cases/case_2/w2c.txt \
  --camera_intrinsics_path test_cases/case_2/intrinsics.txt \
  --prompt_path test_cases/case_2/prompt.txt \
  --save_path test_cases/case_2/output.mp4 \
  --work_dir test_cases/case_2/output \
  --vace_path ./checkpoints/Spatia/step-8500.safetensors \
  --lora_path ./checkpoints/Spatia/lora_weights_10000.safetensors
```

### Case 3

```bash
python inference.py \
  --img_path test_cases/case_3/img.jpg \
  --camera_w2c_path test_cases/case_3/w2c.txt \
  --camera_intrinsics_path test_cases/case_3/intrinsics.txt \
  --prompt_path test_cases/case_3/prompt.txt \
  --mask_path test_cases/case_3/mask.png \
  --save_path test_cases/case_3/output.mp4 \
  --work_dir test_cases/case_3/output \
  --vace_path ./checkpoints/Spatia/step-8500.safetensors \
  --lora_path ./checkpoints/Spatia/lora_weights_10000.safetensors
```

---

## Notes

- `model_weights/Wan2.2-TI2V-5B` must exist before running inference.
- `--vace_path` and `--lora_path` must point to valid Spatia checkpoints.
- Intermediate reconstruction and rendering results are cached in `--work_dir`.
- If prompts, masks, or camera files change, use `--force_rebuild_intermediate` for a clean rerun.

---

## Citation

If you find this project useful, please cite the paper.
```tax
@inproceedings{zhao2026spatia,
  title={Spatia: Video Generation with Updatable Spatial Memory},
  author={Zhao, Jinjing and Wei, Fangyun and Liu, Zhening and Zhang, Hongyang and Xu, Chang and Lu, Yan},
  booktitle={Proceedings of the IEEE/cvf conference on computer vision and pattern recognition},
  year={2026}
}
```


---

<p align="center">
  <small>© 2025 Spatia Project. Licensed under <a href="http://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</small>
</p>
