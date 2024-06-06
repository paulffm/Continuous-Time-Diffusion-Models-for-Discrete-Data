# Vision Transformer

## ViT in General

- ViT

## GenVit; Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model (2022) (Code available)

The first approach to utilize a single ViT as an alternative to the U-Net in DDPM

## DiT: Scalable diffusion models with transformers (2022) (Code available)

- Latent Diffusion Model that replaces the commonly-used U-Net backbone with a transformer that operates on latent patches
- In the paper, they apply DiTs to latent space, although they could be applied to pixel space without modification as well
- This makes their image generation pipeline a hybrid-based approach; they use off-the-shelf convolutional VAEs and transformer-based DDPMs
- VAE as Encoder $E$: $z = E(X)$, then sampling $z$ from Diffusion Model and decoding it to an image $x = D(z)$

### ToDo

- Final layer as logits such that model learns smth? Mabye conv layer
- Unterschied U-Net discrete und continuous

## UViT: All are Worth Words: A ViT Backbone for Diffusion Models (2023) (Code available)

- Latent Diffusion Model with U-ViT architecture as backbone
- predicts noise but can be adapted for $x_0$-prediction
- we find that our U-ViT performs especially well in the latent space, where images are firstly converted to their latent representations before applying diffusion models => must be usable in pixel space

## DiffiT: DIFFUSION VISION TRANSFORMERS FOR IMAGE GENERATION (2024) (No Code)

- We introduce a novel time-dependent self-attention module that allows attention layers to adapt their behavior at different stages of the denoising process in an efficient manner
- We also introduce latent DiffiT which consists of transformer model with the proposed self-attention layers, for high-resolution image generation
- In Image and latent space
- DiffiT uses a symmetrical U-Shaped encoder-decoder architecture in which the contracting and expanding paths are connected to each other via skip connections at every resolution. Specifically, each resolution of the encoder or decoder paths consists of L consecutive DiffiT blocks (Time dependent SA, MLP, Conv ,Skip connections)
- time dependent SA also improves DDPM
- outperforms UViT and DiT

## FiT: Flexible Vision Transformer for Diffusion Model (2024) (No Code)

- transformer architecture specifically de- signed for generating images with unrestricted resolutions and aspect ratios
- FiT conceptualizes images as sequences of dynamically-sized tokens instead of fixed size grids
- One essential network architecture adjustment to handle diverse image sizes is the adoption of 2D Rotary Positional Embedding (RoPE), masked SA

## Idea

To latent space diffusion models in discrete space? Before feeding $x$ into diffusion model
