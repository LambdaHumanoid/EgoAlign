<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/figures/title_dark.svg">
    <img src="assets/figures/title_light.svg" width="700" height="128" alt="EgoAlign — Bridging the Human–Humanoid Gap for Long-Range Loco-Manipulation">
  </picture>
  <p>
    <a href="https://iridescentjiang.github.io/">Yiming Jiang</a><sup>1,4,*</sup>  · 
    <a href="https://jinchen0056.github.io/">Jin Chen</a><sup>2,4</sup>  · 
    <a href="https://chongyang-99.github.io/">Chongyang Xu</a><sup>3,4</sup>  · 
    <a href="https://yilunchen.com/about/">Yilun Chen</a><sup>4,†</sup>  · 
    <a href="https://research.buaa.edu.cn/en/persons/aimin-hao/">Aimin Hao</a><sup>1,‡</sup>  · 
    <a href="https://hyshkust.github.io/">Yisheng He</a><sup>4,†,‡</sup>
  </p>
  <p>
    <sup>1</sup> Beihang University  · 
    <sup>2</sup> Shanghai Innovation Institute &nbsp;·&nbsp;
    <sup>3</sup> Sichuan University  · 
    <sup>4</sup> Alibaba Group
  </p>
  <p>
    <sup>*</sup> Work done during an internship at Alibaba Token Hub (ATH), Alibaba Group.<br>
    <sup>†</sup> Co-project leaders  ·  <sup>‡</sup> Co-corresponding authors.
  </p>
  <p>
    <a href="https://lambdahumanoid.github.io/EgoAlign/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=github" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2609.38046"><img src="https://img.shields.io/badge/PDF-arXiv-red?logo=arxiv" alt="Paper on arXiv"></a>
    <a href="https://lambdahumanoid.github.io/EgoAlign/#full-demo"><img src="https://img.shields.io/badge/Video-Demo-orange" alt="Demo Video"></a>
  </p>
  <p><a href="README.md">English</a> · <a href="README_zh.md">简体中文</a></p>
</div>

<p align="center">
  <strong>Aligning human demonstrations for zero-shot, long-range humanoid loco-manipulation.</strong>
</p>

**EgoAlign** enables **long-range humanoid loco-manipulation** from egocentric human demonstrations. Scale alignment and controller-in-the-loop refinement produce robot-compatible state–action supervision for VLA policies that deploy zero-shot on Unitree G1, without physical-robot task demonstrations.



## 🔄 Pipeline

![EgoAlign pipeline from the paper](assets/figures/pipeline.png)

Human demonstrations are adapted to the robot, refined through simulation, and
converted into paired state–action supervision for VLA fine-tuning.



## ⭐ Key Highlights

- **Human-only demonstrations** for long-range loco-manipulation.
- **Scale alignment and simulation refinement** across embodiments.
- **Causal replay** for paired robot states and motion tokens.
- **Zero-shot G1 deployment** through SONIC whole-body control.



## 🚀 Features & Roadmap

Turn compatible human demonstrations into **G1 robot-learning data**:

- [x] Bilateral palm alignment and SONIC–MuJoCo residual refinement.
- [x] Causal replay with controller history for paired states and motion tokens.
- [x] Object Relocation example, visualization, and endpoint evaluation.
- [x] Additional task comparison videos.
- [ ] VLA training recipes — potential future release; currently refer to [OpenPI](https://github.com/Physical-Intelligence/openpi).

| Processing mode | Description |
|---|---|
| `endpoint-align` | Palm alignment and replay-based refinement |
| `replay-only` | Direct causal replay |

Both export paired **43D states + 76D actions** (64D motion tokens and two 6D hand
commands). See the [processing guide](docs/PROCESSING.md).

> **Alignment options:** `paper` is the default and matches the paper and examples below.
> Add `--alignment-method palm` to preserve palm orientation, with a possible
> **1–2 cm increase in mean endpoint position error**, depending on the task.



## 🛠️ Installation

Use Python 3.10 and start with the [dependency guide](docs/DEPENDENCIES.md).
From the code checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,simulation,data]'

python -m egoalign endpoint-align --help
python -m egoalign replay-only --help
```

### Configure & Verify

1. **Replay environment** — Follow the [runtime setup guide](docs/public_runtime.md) for native or GPU-container execution. Check [dependencies](docs/DEPENDENCIES.md) for Python and CUDA compatibility.
2. **Models & assets** — Configure the human skeleton, BrainCo robot assets, and controller weights using the [asset guide](docs/EXTERNAL_ASSETS.md).
3. **Validation** — Follow the [validation guide](docs/VALIDATION.md) to verify your environment and processed data.



## 📦 Long-Range Loco-Manipulation Example

[Object Relocation](examples/data/basket/) combines long-range locomotion with object manipulation. The video
compares the SMPL reference at physical scale (left), G1 before alignment (middle), and
G1 after alignment (right).

<table align="center"><tr><td align="center" width="590">

https://github.com/user-attachments/assets/3129a65a-8715-4151-a03d-8384ddb4d9a1

<img src="assets/figures/object_relocation_metrics_en.svg" width="100%" alt="Mean palm error ↓ | Before: 14.53 cm | After: 1.71 cm">
</td></tr></table>

After [runtime setup](docs/public_runtime.md), preview the processing commands:

```bash
"$EGOALIGN_DATA_PYTHON" -m egoalign.sample_pipeline basket \
  --raw-root examples/data/basket \
  --robot-xml "$EGOALIGN_RUNTIME_ROOT/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml" \
  --output-root /path/to/new/basket-result --dry-run
```

Review the commands, then follow the [example guide](examples/README.md) to run
the pipeline and export the replay video.



<!-- additional-examples:start -->

## 🎬 More Long-Range Loco-Manipulation Tasks

Left to right: SMPL at physical scale, G1 before alignment, and G1 after alignment (Round 2).

### Storage Box Placement

<table align="center"><tr><td align="center" width="590">

https://github.com/user-attachments/assets/50d807a1-633c-4e5e-ac26-31af12d567a2

<img src="assets/figures/storage_box_metrics_en.svg" width="100%" alt="Mean palm error ↓ | Before: 10.02 cm | After: 1.35 cm">
</td></tr></table>

### Shelf Interaction

<table align="center"><tr><td align="center" width="590">

https://github.com/user-attachments/assets/71dfa4c2-544c-4625-b2d2-8b13f361d30a

<img src="assets/figures/shelf_interaction_metrics_en.svg" width="100%" alt="Mean palm error ↓ | Before: 12.06 cm | After: 2.37 cm">
</td></tr></table>

### Parcel Delivery

<table align="center"><tr><td align="center" width="590">

https://github.com/user-attachments/assets/edc0d6e7-979d-4917-bff6-0a12ba669ecc

<img src="assets/figures/parcel_delivery_metrics_en.svg" width="100%" alt="Mean palm error ↓ | Before: 13.86 cm | After: 4.45 cm">
</td></tr></table>

### Indoor Cleaning

<table align="center"><tr><td align="center" width="590">

https://github.com/user-attachments/assets/198caafd-7a0f-48e2-b434-868fee009aa2

<img src="assets/figures/indoor_cleaning_metrics_en.svg" width="100%" alt="Mean palm error ↓ | Before: 12.49 cm | After: 2.37 cm">
</td></tr></table>

### Measurement Protocol

Videos preserve metric world motion. Human fingers visualize the same grasp commands as the after-alignment robot.

All results use a fixed **0.24 s delay compensation**.

| Item         | Definition                                                                    |
| ------------ | ----------------------------------------------------------------------------- |
| Error        | Mean Euclidean distance over both palms and all valid frames; lower is better |
| Time pairing | Target frame i vs. robot frame i+12 at 50 Hz; final 12 target frames omitted  |
| Coordinates  | Gravity-aligned heading frame; pelvis-centered XY; lower-foot-grounded Z                                     |
| Video timing | Original source clock, without delay compensation                             |

See the [measurement record](assets/metrics/media_manifest.json) for full precision and quality checks.

<!-- additional-examples:end -->

## 📁 Code Structure

```text
egoalign/             Command entry points and pipeline orchestration
  geometry/          Coordinate and rotation math
  robots/g1/         G1 model, joint metadata and BrainCo hand mapping
  datasets/          Data schemas, export and video writing
  adapters/sonic/    SONIC state, rollout and telemetry contracts
scripts/human_data/   Alignment, residual refinement, and data export
runtime/             Public-runtime patches and source manifests
configs/             Dependency versions and asset hashes
examples/            Object-relocation input and reproduction instructions
tools/               Runtime setup, replay adapter and validation tools
  run_sim_replay.sh  Simulation-replay adapter used by the pipelines
tests/               Automated tests
docs/                Processing, runtime, data-contract, and validation guides
assets/              Paper figure, task videos, and measurement records
```

## 📘 Documentation

See the [project website](https://lambdahumanoid.github.io/EgoAlign/) for the paper and demo.

- [Processing workflow and commands](docs/PROCESSING.md)
- [Input/output data contract](docs/DATA_CONTRACT.md)
- [Public SONIC–MuJoCo runtime setup](docs/public_runtime.md)
- [External assets](docs/EXTERNAL_ASSETS.md) and [dependencies](docs/DEPENDENCIES.md)
- [Sample reproduction and verification](examples/README.md)
- [Validation guide](docs/VALIDATION.md)
- [Third-party components and terms](docs/THIRD_PARTY.md)

For development checks:

```bash
python -m pytest -q
python tools/check_cli.py
python tools/audit_source.py
```

## 📚 Citation

If you use EgoAlign in your research, please cite our paper:

```bibtex
@misc{jiang2026egoalign,
  title = {{EgoAlign}: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation},
  author = {Yiming Jiang and Jin Chen and Chongyang Xu and
            Yilun Chen and Aimin Hao and Yisheng He},
  year = {2026},
  eprint = {2609.38046},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.38046}
}
```

## 🙏 Acknowledgments

We thank the teams behind these open-source projects:

| [![OpenPI](https://img.shields.io/badge/OpenPI-Physical%20Intelligence-d58b45)](https://github.com/Physical-Intelligence/openpi) | [![SONIC](https://img.shields.io/badge/SONIC-NVIDIA-76b900)](https://github.com/NVlabs/GR00T-WholeBodyControl) | [![XRoboToolkit](https://img.shields.io/badge/XRoboToolkit-PICO%20VR-397bbf)](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client) | [![Open GoPro](https://img.shields.io/badge/Open%20GoPro-GoPro-009fdf)](https://github.com/gopro/OpenGoPro) |
| :---: | :---: | :---: | :---: |
| VLA&nbsp;models | Whole-body&nbsp;control | PICO&nbsp;tracking | GoPro&nbsp;interface |

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).

## 📬 Contact

Feel free to visit my [homepage](https://iridescentjiang.github.io/). You can reach me at [jiangyimingjym@buaa.edu.cn](mailto:jiangyimingjym@buaa.edu.cn).
