# ZIA: Dual-ViT Framework for Data Valuation and Multi-Label X-Ray Image Classification

This repository contains the code and experiments for the paper **"ZIA: Dual-ViT Framework for Data Valuation and Multi-Label X-Ray Image Classification"**. The proposed framework combines a lightweight Vision Transformer model with a data valuation module to enhance training efficiency in medical image analysis.

---

## 🧩 Overview

- ⚕️ **Task:** Multi-label classification of chest X-ray images using a Vision Transformer (ViT-Tiny).

- 🧠 **Contribution:** Introduces a dual-path framework where one branch handles classification and the other estimates sample informativeness based on loss profiles and clustering.

- 📊 **Goal:** Improve model performance while training on a compact, high-value subset of data.

---

## ✨ Features

- ✅ Vision Transformer-based classifier for grayscale medical images.

- 📉 Loss profiling to assess sample difficulty.

- 🔗 GMM-based clustering for unsupervised data valuation.

- 🔁 Subset selection to reduce training data while preserving classification accuracy.

---

## 🔬 Applications

- 🔍 Efficient training on large-scale medical datasets like **MIMIC-CXR**.

- 🧪 Identification of high-value samples for model fine-tuning.

- 💾 Reduction of training time and memory consumption in medical imaging pipelines.
