<img src="./gigagan-sample.png" width="500px"></img>

<img src="./gigagan-architecture.png" width="500px"></img>

## GigaGAN - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2303.05511v2">GigaGAN</a> <a href="https://mingukkang.github.io/GigaGAN/">(project page)</a>, new SOTA GAN out of Adobe.

I will also add a few findings from <a href="https://github.com/lucidrains/lightweight-gan">lightweight gan</a>, for faster convergence (skip layer excitation), better stability (reconstruction auxiliary loss in discriminator), as well as improved results (GLU in generator).

It will also contain the code for the 1k - 4k upsamplers, which I find to be the highlight of this paper.

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in helping out with the replication with the <a href="https://laion.ai/">LAION</a> community

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their accelerate library

- All the maintainers at <a href="https://github.com/mlfoundations/open_clip">OpenClip</a>, for their SOTA open sourced contrastive learning text-image models

- <a href="https://github.com/XavierXiao">Xavier</a> for reviewing the discriminator code and pointing out that the scale invariance was not correctly built!

## Usage

Simple unconditional GAN, for starters

```python
import torch

from gigagan_pytorch import (
    GigaGAN,
    ImageDataset
)

gan = GigaGAN(
    generator = dict(
        dim = 64,
        style_network = dict(
            dim = 64,
            depth = 4
        ),
        image_size = 256,
        dim_max = 512,
        use_glu = True,
        num_skip_layers_excite = 4,
        unconditional = True
    ),
    discriminator = dict(
        dim = 64,
        dim_max = 512,
        image_size = 256,
        use_glu = True,
        num_skip_layers_excite = 4,
        unconditional = True
    )
).cuda()

# dataset

dataset = ImageDataset(
    folder = '/path/to/your/data',
    image_size = 256
)

dataloader = dataset.get_dataloader(batch_size = 1)

# training the discriminator and generator alternating
# for 100 steps in this example, batch size 1, gradient accumulated 8 times

gan(
    dataloader = dataloader,
    steps = 100,
    grad_accum_every = 8
)
```

## Todo

- [x] make sure it can be trained unconditionally
- [x] read the relevant papers and knock out all 3 auxiliary losses
    - [x] matching aware loss
    - [x] clip loss
    - [x] vision-aided discriminator loss
    - [x] add reconstruction losses on arbitrary stages in the discriminator (lightweight gan)
    - [x] figure out how the random projections are used from projected-gan
    - [x] vision aided discriminator needs to extract N layers from the vision model in CLIP
    - [x] figure out whether to discard CLS token and reshape into image dimensions for convolution, or stick with attention and condition with adaptive layernorm - also turn off vision aided gan in unconditional case
- [x] unet upsampler
    - [x] add adaptive conv
    - [x] modify latter stage of unet to also output rgb residuals, and pass the rgb into discriminator. make discriminator agnostic to rgb being passed in
    - [x] do pixel shuffle upsamples for unet
- [x] get a code review for the multi-scale inputs and outputs, as the paper was a bit vague
- [ ] do a review of the auxiliary losses
- [ ] add upsampling network architecture
- [ ] port over CLI from lightweight|stylegan2-pytorch
- [ ] hook up laion dataset for text-image


## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.05511,
    url     = {https://arxiv.org/abs/2303.05511},
    author  = {Kang, Minguk and Zhu, Jun-Yan and Zhang, Richard and Park, Jaesik and Shechtman, Eli and Paris, Sylvain and Park, Taesung},  
    title   = {Scaling up GANs for Text-to-Image Synthesis},
    publisher = {arXiv},
    year    = {2023},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@article{Liu2021TowardsFA,
    title   = {Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis},
    author  = {Bingchen Liu and Yizhe Zhu and Kunpeng Song and A. Elgammal},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2101.04775}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```
