# Dual Discriminator GAN with self attention

This project was implemented as part of the CS787: Generative AI course at IIT Kanpur during my First Semester of M.Tech, 2025.

This project evaluates four generative models on the CIFAR-10 dataset:

1. **DCGAN**  
2. **DCGAN + Self-Attention (SAGAN-style)**
3. **Dual-Discriminator GAN (D2GAN)**
4. **Dual-Discriminator GAN + Self-Attention**

The final goal was to integrate self attention mechanism in a dual discriminator GAN, improving diversity as well as quality.

---

## Models

### 1. DCGAN  
Baseline convolutional GAN using transposed convolutions for upsampling and LeakyReLU in the discriminator.

### 2. DCGAN with Self-Attention  
Adds a self-attention block to capture long-range spatial dependencies and improve global consistency.

### 3. D2GAN  
Uses two discriminators:
- **D1** encourages diversity (forward KL)
- **D2** encourages realism (reverse KL)

This helps reduce mode collapse and encourages broader coverage of the data distribution.

### 4. D2GAN with Self-Attention  
Combines both ideas:
- dual-discriminator loss  
- attention-based feature modelling  
This is the most expressive model tested.

---

## Training
All models are trained on CIFAR-10 (32×32 RGB).  
Training scripts include:
- checkpointing
- deterministic mode (optional)
- fixed-noise visualisation
- periodic sample generation

---

## Evaluation
We compute:
- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**
Generated samples and metrics are stored every 5 epochs.


---
