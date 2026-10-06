<div align="center">
  <h1>EgoAlign: Bridging the Human–Humanoid Gap<br>for Long-Range Loco-Manipulation</h1>
  <p>
    <a href="https://iridescentjiang.github.io/">Yiming Jiang</a><sup>1,4,*</sup> &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=4FqHXOsAAAAJ">Jin Chen</a><sup>2,4</sup> &nbsp;·&nbsp;
    <a href="https://chongyang-99.github.io/">Chongyang Xu</a><sup>3,4</sup> &nbsp;·&nbsp;
    <a href="https://yilunchen.com/about/">Yilun Chen</a><sup>4,†</sup> &nbsp;·&nbsp;
    <a href="https://research.buaa.edu.cn/en/persons/aimin-hao/">Aimin Hao</a><sup>1,‡</sup> &nbsp;·&nbsp;
    <a href="https://hyshkust.github.io/">Yisheng He</a><sup>4,†,‡</sup>
  </p>
  <p>
    <sup>1</sup> Beihang University &nbsp;·&nbsp;
    <sup>2</sup> Shanghai Innovation Institute<br>
    <sup>3</sup> Sichuan University &nbsp;·&nbsp;
    <sup>4</sup> Alibaba Group
  </p>
  <p>
    <sup>*</sup> Work done during an internship at Alibaba Token Hub (ATH), Alibaba Group.<br>
    <sup>†</sup> Co-project leaders &nbsp;·&nbsp; <sup>‡</sup> Co-corresponding authors.
  </p>
  <p>
    <a href="https://lambdahumanoid.github.io/EgoAlign/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=github" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2609.38046"><img src="https://img.shields.io/badge/PDF-arXiv-red?logo=arxiv" alt="Paper on arXiv"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License: Apache 2.0"></a>
  </p>
</div>

## 📖 Abstract

Long-range loco-manipulation requires a humanoid to coordinate walking, reaching, and interacting across a scene. Human demonstrations provide a natural source of task experience, but differences in body proportions and controller response make these motions difficult to transfer directly to a robot.

We present **EgoAlign**, a framework that transforms egocentric human demonstrations into robot-compatible action and state supervision. We capture egocentric views, body motion, and hand commands with real-time dynamics-guided feedback, then adapt the recorded motion through kinematic scale alignment and controller-in-the-loop refinement. Causal replay reconstructs the corresponding robot states and action targets for vision-language-action (VLA) fine-tuning.

Using human task demonstrations alone, the fine-tuned policy deploys zero-shot on a Unitree G1 through SONIC's whole-body control interface. It performs long-range object relocation, position-generalized navigation, and foot interaction without physical-robot task demonstrations.

## ⭐ Key Highlights

1. **Human-only task training.** Learn long-range humanoid behaviors from egocentric human demonstrations, without collecting physical-robot demonstrations of the target tasks.
2. **Alignment across embodiment and dynamics.** Kinematic scale alignment adapts upper-body interaction geometry while preserving global travel references; controller-in-the-loop refinement accounts for the robot's realized motion.
3. **Robot-compatible supervision.** Causal replay pairs pre-action robot states with same-tick motion tokens from the final adapted rollout, connecting human demonstrations to the robot's control interface.
4. **Zero-shot physical deployment.** A fine-tuned VLA predicts motion tokens that SONIC decodes using live robot state history, enabling closed-loop loco-manipulation on a Unitree G1.

## 🚀 Code Release

**Code will be released here in the future.** This repository currently contains the project website and its media assets in [`docs/`](docs/). Visit the [project page](https://lambdahumanoid.github.io/EgoAlign/) for demonstrations and the method overview.

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

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).
