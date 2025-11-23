# Dino-Mamba: Visual Mamba with DINO Self-Supervised Learning

Official PyTorch implementation of **"RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models"**

**Authors:** Timing Yang, Feng Wang, Guoyizhe Wei, Alan Yuille  
Johns Hopkins University

[[Paper]](https://arxiv.org/abs/XXXX.XXXXX)


## Overview

Mamba, originally introduced for language modeling, has recently garnered attention as an effective backbone for vision tasks. However, its underlying mechanism in visual domains remains poorly understood. In this work, we systematically investigate Mamba’s representational properties and make three primary contributions. First, we theoretically analyze Mamba’s relationship to Softmax and Linear Attention, confirming that it can be viewed as a low-rank approximation of Softmax Attention and thereby bridging the representational gap between Softmax and Linear forms. Second, we introduce a novel binary segmentation metric for activation map evaluation, extending qualitative assessments to a quantitative measure that demonstrates Mamba’s capacity to model long-range dependencies. Third, by leveraging DINO for self-supervised pretraining, we obtain clearer activation maps than those produced by standard supervised approaches, highlighting Mamba’s potential for interpretability. Notably, our model also achieves a 78.5% linear probing accuracy on ImageNet, underscoring its strong performance. We hope this work can provide valuable insights for future investigations of Mamba-based vision architectures.

<p align="center">
  <img src="./img/Vis-attn-linearattn-mamba.png" width="900">
  <br>
  <em>Feature map comparison: Self-Attention produces the clearest activation maps with strong foreground-background distinction, Mamba shows similar patterns with moderate noise, while Linear Attention struggles to focus on informative regions.</em>
</p>




---

## Model Zoo

| Model | Dimension | ImageNet Top-1 | Download |
|-------|-----------|----------------|----------|
| DinoVim-Tiny | 256 | 73.7% | [DinoVim-Tiny](https://huggingface.co/Timing1/Dino-Mamba/resolve/main/Vim/Tiny/) |
| DinoVim-Small | 512 | 77.4% | [DinoVim-Small](https://huggingface.co/Timing1/Dino-Mamba/resolve/main/Vim/Small/) |
| DinoVim-Base | 768 | 78.1% | [DinoVim-Base](https://huggingface.co/Timing1/Dino-Mamba/resolve/main/Vim/Base/) |
| DinoMamba-Reg-Base | 768 | **78.5%** | [DinoMamba-Reg-Base](https://huggingface.co/Timing1/Dino-Mamba/resolve/main/Mambar/Base/) |

##  Requirements

```bash
torch==2.0.0+cu118 
torchvision==0.15.1+cu118 
timm==0.4.12 
mamba-ssm==2.0.4
causal-conv1d==1.2.1
```

## Training

### Self-Supervised Pretraining with DINO

**Vim Models:**
```bash
# Vim-Tiny
torchrun --nproc_per_node=4 --master_port=15100 main_dino_vim.py \
    --arch vim_tiny_patch16_224 \
    --data_path /path/to/imagenet/train \
    --output_dir ./output_vim_tiny/

# Vim-Small
torchrun --nproc_per_node=4 --master_port=15100 main_dino_vim.py \
    --arch vim_small_patch16_224 \
    --data_path /path/to/imagenet/train \
    --output_dir ./output_vim_small/

# Vim-Base
torchrun --nproc_per_node=4 --master_port=15100 main_dino_vim.py \
    --arch vim_base_patch16_224 \
    --data_path /path/to/imagenet/train \
    --output_dir ./output_vim_base/
```

**Mamba-Reg Models:**
```bash
# MambaReg-Tiny
torchrun --nproc_per_node=4 --master_port=15100 main_dino_mambar.py \
    --arch mambar_tiny_patch16_224 \
    --data_path /path/to/imagenet/train \
    --output_dir ./output_mambar_tiny/

# MambaReg-Small
torchrun --nproc_per_node=4 --master_port=15100 main_dino_mambar.py \
    --arch mambar_small_patch16_224 \
    --data_path /path/to/imagenet/train \
    --output_dir ./output_mambar_small/

# MambaReg-Base
torchrun --nproc_per_node=4 --master_port=15100 main_dino_mambar.py \
    --arch mambar_base_patch16_224 \
    --data_path /path/to/imagenet/train \
    --output_dir ./output_mambar_base/
```

## Evaluation

### Linear Probing on ImageNet

**Vim Models:**
```bash
# Vim-Tiny
torchrun --nproc_per_node=4 --master_port=15400 eval_linear_vim.py \
    --arch vim_tiny_patch16_224 \
    --pretrained_weights ./output_vim_tiny/checkpoint.pth \
    --data_path /path/to/imagenet \
    --output_dir ./output_vim_tiny/ \
    --n_last_blocks 8 \
    --evaluate

# Vim-Small
torchrun --nproc_per_node=4 --master_port=15400 eval_linear_vim.py \
    --arch vim_small_patch16_224 \
    --wd 0.001 \
    --pretrained_weights ./output_vim_small/checkpoint.pth \
    --data_path /path/to/imagenet \
    --output_dir ./output_vim_small/ \
    --n_last_blocks 4 \
    --evaluate

# Vim-Base
torchrun --nproc_per_node=4 --master_port=15400 eval_linear_vim.py \
    --arch vim_base_patch16_224 \
    --wd 0.01 \
    --pretrained_weights ./output_vim_base/checkpoint.pth \
    --data_path /path/to/imagenet \
    --output_dir ./output_vim_base/ \
    --n_last_blocks 8 \
    --evaluate
```

**Mamba-Reg Models:**
```bash
# MambaReg-Base
torchrun --nproc_per_node=4 --master_port=15400 eval_linear_mambar.py \
    --arch mambar_base_patch16_224 \
    --wd 0.01 \
    --pretrained_weights ./output_mambar_base/checkpoint.pth \
    --data_path /path/to/imagenet \
    --output_dir ./output_mambar_base/ \
    --n_last_blocks 8 \
    --evaluate
```

## Citation

If you find this work useful, please cite:
```bibtex
@article{yang2025mamba,
  title={RNN as Linear Transformer: A Closer Investigation into Representational Potentials of Visual Mamba Models},
  author={Yang, Timing and Wang, Feng and Wei, Guoyizhe and Yuille, Alan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```



## Acknowledgements

This work builds upon [DINO](https://github.com/facebookresearch/dino), [Mamba](https://github.com/state-spaces/mamba), [Mamba-Reg](https://github.com/wangf3014/Mamba-Reg) , and [Vision Mamba](https://github.com/hustvl/Vim).

## License

This project is released under the Apache 2.0 license.


