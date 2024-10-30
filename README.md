
# TokenFormer: a fully attention-based neural network with tokenized model parameters. Maximizing the flexibility of Transformer.
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.09394-b31b1b.svg?logo=arXiv)](https://arxiv.org/)
[![project page](https://img.shields.io/badge/%F0%9F%A4%97%20ProjectPages-TokenFormer-red)](https://haiyang-w.github.io/tokenformer.github.io/)
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-TokenFormer-yellow)](https://huggingface.co/Haiyang-W)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FHaiyang-W%2FTokenFormer%2Ftree%2Fmain&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
</h5>

This repo is the official implementation of our paper: [TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters](https://arxiv.org/) as well as the follow-ups. Our TokenFormer is a natively scalable architecture that leverages the attention mechanism not only for computations among input tokens but also for interactions between tokens and model parameters, thereby enhancing architectural flexibility. We have made every effort to ensure that the codebase is clean, concise, easily readable, state-of-the-art, and relies only on minimal dependencies.

> TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters
>
> [Haiyang Wang](https://haiyang-w.github.io/), [Yue Fan](https://yue-fan.github.io/), [Muhammad Ferjad Naeem](https://ferjad.github.io/), [Yongqin Xian](https://xianyongqin.github.io/), [Jan Eric Lenssen](https://janericlenssen.github.io/), [Liwei Wang](http://www.liweiwang-pku.com/), [Federico Tombari](https://federicotombari.github.io/), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele)
> - Primary contact: Haiyang Wang (haiwang@mpi-inf.mpg.de), Bernt Schiele (schiele@mpi-inf.mpg.de)

<div align="center">
  <img src="assets/Figure1.png" width="800"/>
</div>

## 📣 News
- [24-10-30] 🚀 Inference code is released.
- [24-10-30] 👀 TokenFormer is released on [arXiv](https://arxiv.org/).

## Overview
- [💫 What we want to do](https://github.com/Haiyang-W/TokenFormer#what-we-want-to-do)
- [🚀 Main Results](https://github.com/Haiyang-W/TokenFormer#main-results)
- [📘 Model Zoo](https://github.com/Haiyang-W/TokenFormer#model-zoo)
- [🛠️ Quick Start](https://github.com/Haiyang-W/TokenFormer#quick-start)
- [👀 Todo](https://github.com/Haiyang-W/TokenFormer#todo)
- [📘 Citation](https://github.com/Haiyang-W/TokenFormer#citation)

## 💫 What want to do?
We introduce Tokenformer, a <font color="red">**fully attention-based**</font> architecture that unifies the computations of token-token and token-parameter interactions by entirely employing the attention mechanism,  <font color="red">**maximizes the flexibility of neural network**</font>. The advantage makes it can handle a variable number of parameters, inherently enhances the model's scalability, facilitating progressively efficient scaling.

<font color="red">**We not only tokenizes data but also model parameters, replacing the model concept with interaction flows between data and parameter tokens, further advancing the network architecture towards unification.**</font>

Hope that this architecture can offer greater flexibility than traditional Transformers, will further contribute to the development of *foundation models*, *sparse inference (MoE)*, *parameter efficient tuning*, *device-cloud collaboration*, *vision-language*,  *model interpretability*, and so on.

```
# Pattention Implementations with given inputs

query, key, value = inputs, key_param_tokens, value_param_tokens

attn_weight = query @ key.transpose(-2, -1) * scale_factor

attn_weight *= attn_masks
# modified softmax, softmax is equal to exp + L1 norm
attn_weight = nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)

output = attn_weight @ value
```

## 🚀 Main results
### Incremental model scaling
<div align="center">
  <img src="assets/Figure2.png" width="800"/>
</div>

Traditionally, large transformer architectures are trained from scratch without reusing previous smaller-scale models. In this paper, we propose a novel fully attention-based architecture that allows scaling model incrementally, thus greatly reducing the overall cost of training large transformer architectures.

### Language modeling on Pile dataset with zero-shot evaluation

(**Zero-shot Evaluations.**) The best performance for each model size is highlighted in bold. Our comparisons are made with publicly available transformer-based LMs with various tokenizers. Following Pythia, our model is trained for up to 300B tokens on pile dataset.

<div align="center">
  <img src="assets/Figure3.png" width="800"/>
</div>


### Visual modeling on ImageNet-1k classification

(**Image Classification.**) Comparison of standard vision transformer on ImageNet-1K.

<div align="center">
  <img src="assets/Figure4.png" width="1000"/>
</div>


## 📘 Model Zoo
### Language Modeling Benchmark (Pile)

Pretrained models are uploaded to [huggingface](https://huggingface.co/Haiyang-W) ``TokenFormer-150M``, ``TokenFormer-450M``, ``TokenFormer-900M`` and ``TokenFormer-1-5B``, trained on 300B tokens on the Pile.

These models were trained on the [Pile](https://huggingface.co/datasets/EleutherAI/pile), and follow the standard model dimensions of Transformer, and evaluated on standard zero-shot benchmark described by mamba:
|  Model  |Params| Layers | Model dim. |ckpt|config|
|---------|---------|---------|--------|--------|---------|
|  TokenFormer-150M | 150M | 12 | 768  |[ckpt](https://huggingface.co/Haiyang-W/TokenFormer-150M/tree/main)| [config](https://github.com/Haiyang-W/TokenFormer/blob/main/configs/tokenformer/150M_eval.yml) |
|  TokenFormer-450M | 450M | 24 | 1024 |[ckpt](https://huggingface.co/Haiyang-W/TokenFormer-450M/tree/main)| [config](https://github.com/Haiyang-W/TokenFormer/blob/main/configs/tokenformer/450M_eval.yml) |
|  TokenFormer-900M| 900M| 32 | 1280 |[ckpt](https://huggingface.co/Haiyang-W/TokenFormer-900M/tree/main)| [config](https://github.com/Haiyang-W/TokenFormer/blob/main/configs/tokenformer/900M_eval.yml) |
|  TokenFormer-1-5B| 1-5B| 40 | 1536 |[ckpt](https://huggingface.co/Haiyang-W/TokenFormer-1-5B/tree/main)| [config](https://github.com/Haiyang-W/TokenFormer/blob/main/configs/tokenformer/1-5B_eval.yml) |

Note: these are base models trained only for 300B tokens, without any form of downstream modification (instruction tuning, etc.). Performance is expected to be comparable or better than other architectures trained on similar data, but not to match larger or fine-tuned models.

### Visual Modeling Benchmark (DataComp-1B on CLIP approach)
Will be released later.

## 🛠️ Quick Start
### Installation
First make sure you are in an environment with Python 3.8 with an appropriate version of PyTorch 1.8 or later installed. **Note:** our TokenFormer is based on the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), some of the libraries that GPT-NeoX depends on have not been updated to be compatible with Python 3.10+. Python 3.9 appears to work, but this codebase has been developed and tested for Python 3.8.

To install the remaining basic dependencies, run:
```
conda create -n TokenFormer python=3.8

git clone https://github.com/Haiyang-W/TokenFormer.git

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# raven module load gcc/10

pip install -r requirements/requirements.txt
pip install -r requirements/requirements-flashattention.txt # need gcc > 9
pip install -r requirements/requirements-wandb.txt # optional, if logging using WandB
pip install -r requirements/requirements-tensorboard.txt # optional, if logging via tensorboard
pip install -r requirements/requirements-comet.txt # optional, if logging via Comet

# install apex
pip install -r requirements/requirements-apex-pip.txt # pip > 23.1
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
from the repository root.

### Evaluations
To run zero-shot evaluations of models (corresponding to Table 1 of the paper), we use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library.

First you should download the pre-trained weights from [huggingface](https://huggingface.co/Haiyang-W) to your local directory. For example, the relative path is ``./TokenFormer-150M/pytorch_model.bin`` from the repository root. 
```
# single-gpu evaluation (currently only tested on single-gpu.)

cd ./TokenFormer
python ./deepy.py eval.py -d configs tokenformer/150M_eval.yml --eval_tasks lambada_openai hellaswag piqa arc_challenge arc_easy winogrande
```

## 👀 TODO

- [x] Release the [arXiv](https://arxiv.org/abs/2301.06051) version.
- [x] Release inference code and model weights of LLM.
- [ ] Release training code of LLM.
- [ ] Release incremental scaling training code of LLM.
- [ ] Release training code of Image Classification.
- [ ] Release model weights of CLIP trained on DataComp-1B.
- [ ] Release some initial results of Vision Language Modeling on LLaVA benchmark.

## 📘 Citation
Please consider citing our work as follows if it is helpful.
```
```
