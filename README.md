<div align="center">

<h1>VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion and Restoration (CVPR 2026)</h1>

<p>
<b>Official PyTorch implementation</b> of <i>"VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion and Restoration"</i>
</p>

<p>
<a href="(Paper link)"><img src="https://img.shields.io/badge/Paper-CVPR%202026-blue"></a>
<a href="(arXiv link)"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b"></a>
<a href="(Project page)"><img src="https://img.shields.io/badge/Project-Page-green"></a>
<a href="(Dataset link)"><img src="https://img.shields.io/badge/Dataset-M3SVD-orange"></a>
</p>

<p>
Linfeng Tang, Yeda Wang, Meiqi Gong, Zizhuo Li, Yuxin Deng, Xunpeng Yi, Chunyu Li, Hao Zhang, Han Xu, Jiayi Ma
</p>

</div>

---

## 🔥 News
- **[2026]** VideoFusion has been **accepted to CVPR 2026**.
- We release **M3SVD**, a large-scale aligned **infrared-visible multi-modal video dataset** for fusion & restoration.
- Pretrained weights / dataset links / scripts will be updated here.

---

## 🔎 Overview

### Motivation
Most multi-modal fusion methods are designed for **static images**. Applying them frame-by-frame to videos often leads to:
- **Temporal flickering** (inconsistent fusion across frames)
- Under-utilization of **motion/temporal cues**
- Poor robustness under **real-world degradations**

VideoFusion explicitly models **cross-modal complementarity + temporal dynamics**, and supports **multi-modal video fusion & restoration** in a unified framework.

<p align="center">
  <img src="assets/Motivation.jpg" width="90%">
</p>

---

## ✨ Key Contributions
- **VideoFusion**: a spatio-temporal collaborative network for **multi-modal video fusion and restoration**.
- **Spatio-temporal collaboration**:
  - cross-modal differential reinforcement
  - complete modality-guided fusion
  - bi-temporal co-attention (forward/backward temporal aggregation)
- **Temporal stabilization** via **variation-level consistency constraint** to reduce flicker.
- **M3SVD dataset**: a large-scale synchronized & registered IR-VI video benchmark.

---

## 🧠 Method at a Glance

### Architecture
Given aligned IR/VI video clips, VideoFusion outputs:
- **Fused video** (RGB)
- **Restored IR & VI** (auxiliary modality-unmixing branch for better fusion and robustness)

Core modules:
1. **CmDRM**: Cross-modal Differential Reinforcement Module  
2. **CMGF**: Complete Modality-guided Fusion  
3. **BiCAM**: Bi-temporal Co-attention Module  
4. **Transformer-based enhancement** (Restormer-style operator)
5. **Modality Unmixing** + IR/VI decoders

<p align="center">
  <img src="assets/Framework.jpg" width="92%">
</p>

---

## 📦 M3SVD Dataset

### Scale & Properties
- **220** temporally synchronized & spatially registered IR-VI videos  
- **153,797** frames total  
- Registered resolution **640×480**, **30 FPS**  
- Diverse conditions: **daytime / nighttime / challenging scenarios** (e.g., occlusion, disguise, low illumination, overexposure)

<p align="center">
  <img src="assets/Datasets.jpg" width="92%">
</p>

### Dataset Comparison (vs. prior works)
<p align="center">
  <img src="assets/Dataset_Comparison.jpg" width="92%">
</p>

### Acquisition & Registration (Brief)
- Synchronized dual-spectral capture (IR + Visible)
- Distortion calibration (both sensors)
- Estimate robust multimodal correspondences and compute homography for spatial registration

<p align="center">
  <img src="assets/Device.jpg" width="92%">
</p>

> 📌 Place dataset files following the dataloader requirement (see **Dataset Preparation** section).  
> 🔗 Download links will be updated: **(TBD)**

---

## 🖼️ Qualitative Results

### Fusion Quality (examples)
<p align="center">
  <img src="assets/Qualitative.jpg" width="92%">
</p>

### Restoration / Robustness under Degradations
<p align="center">
  <img src="assets/Restoration.jpg" width="92%">
</p>

### Challenging Scenarios
<p align="center">
  <img src="assets/Challenging.jpg" width="92%">
</p>

---

## ⏱️ Temporal Consistency

VideoFusion emphasizes temporal coherence. We provide temporal visualization examples:

<p align="center">
  <img src="assets/Temporal.jpg" width="92%">
</p>

<p align="center">
  <img src="assets/Frames.jpg" width="92%">
</p>

---

## 📈 Ablation & Analysis

### Ablation Study
<p align="center">
  <img src="assets/Ablation.jpg" width="92%">
</p>

### Loss Curves
<p align="center">
  <img src="assets/loss.jpg" width="70%">
</p>

### Radar Charts (metric summary)
<p align="center">
  <img src="assets/Radar.jpg" width="46%">
  <img src="assets/Radar_zoom.jpg" width="46%">
</p>

---

## 🎯 Downstream / Tracking Demo (Optional)
<p align="center">
  <img src="assets/Track.jpg" width="92%">
</p>

---

## 🧩 Pipeline / Workflow
<p align="center">
  <img src="assets/Workflow.jpg" width="92%">
</p>

---

## ⚙️ Installation

### 1) Clone
```bash
git clone (your_repo_url_here)
cd VideoFusion
