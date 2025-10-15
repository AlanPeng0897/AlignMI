<h1 align="center">Generative Model Inversion Through the Lens of the Manifold Hypothesis</h1>
<p align="center">
    <a href="https://github.com/AlanPeng0897/AlignMI"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://arxiv.org/pdf/2509.20177"> <img src="https://img.shields.io/badge/NeurIPS'25-blue" alt="License"> </a>
</p>

Hi, this is the code for our NeurIPS 2025 paper: *[Generative Model Inversion Through the Lens of the Manifold Hypothesis](https://arxiv.org/pdf/2509.20177)*.
This repository provides tools to empirically validate gradient–manifold alignment hypotheses and train alignment-aware models for improved model inversion.

---

## 📦 Environment Setup

Install all required dependencies using the provided environment file:

```bash
conda env create -f AlignMI.yaml
conda activate AlignMI
```

Visit the shared Google Drive folder:  [👉Pretrained models](https://drive.google.com/drive/folders/1fPSoQrMzwohgkqdLJ9EwdEkgfTtft2rV?usp=sharing)

Download the contents and place them into the project directory. 

## 🔍 Empirical Validation of the Hypothesis
### 🧠 Tangent-Space Basis Computation

This step encodes input images using a pretrained VAE and computes the tangent-space basis of the data manifold via JVP + SVD. The results are saved as `(x, y, U)` tuples for downstream analysis.

### Usage

**Single-process (rank 0 of 1):**
```bash
python compute_tangent_space_basis.py \
  --config ./configs/training/targets/compute_tangent_space_basis.yaml \
  --output_dir ./tangent_space \
  --batch_size 100 \
  --chunk_size 8 \
  --world_size 1 \
  --rank 0
```

**Multi-GPU example:**
```bash
for RANK in $(seq 0 $((WORLD_SIZE-1))); do
  CUDA_VISIBLE_DEVICES=$RANK python compute_tangent_space_basis.py \
    --config ./configs/training/targets/compute_tangent_space_basis.yaml \
    --output_dir ./tangent_space \
    --batch_size 100 \
    --chunk_size 8 \
    --world_size 10 \
    --rank $RANK &
done

```



### 🧩 Training the Alignment-Aware Model

Assuming your tangent-space files (e.g., `x_y_U_list_subset0.pt`) are ready, launch the alignment-aware training with:

```bash
python train_align_model.py \
  --config ./configs/training/targets/vgg16_align_train.yaml
```



## 🔍 Evaluation of Gradient–Manifold Alignment (AlignMI)

### ➤ Baseline (Standard GMI)
```bash
CUDA_VISIBLE_DEVICES=0 python attack_gmi.py -sg \
  --exp_name celeba_vgg16_gmi_id0-100 \
  --config configs/attacking/gmi_stylegan-celeba_vgg16-celeba.yaml
```

### ➤ PAA (Perturbation-Averaged Alignment)
```bash
CUDA_VISIBLE_DEVICES=0 python attack_gmi.py -sg \
  --exp_name celeba_vgg16_gmi_id0-100 \
  --config configs/attacking/gmi_stylegan-celeba_vgg16-celeba.yaml
```

### ➤ TAA (Transformation-Averaged Alignment)
```bash
CUDA_VISIBLE_DEVICES=0 python attack_gmi.py -sg \
  --exp_name celeba_vgg16_gmi_id0-100 \
  --config configs/attacking/gmi_stylegan-celeba_vgg16-celeba.yaml
```




## References
If you find this code helpful in your research, please consider citing
```bibtex
@inproceedings{peng2025AlignMI,
title={Generative Model Inversion Through the Lens of the Manifold Hypothesis},
author={Peng, Xiong and Han, Bo and Yu, Fengfei and Liu, Tongliang and Liu, Feng and Zhou, Mingyuan},
booktitle={NeurIPS},
year={2025}
}
```