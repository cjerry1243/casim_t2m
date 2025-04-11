# CASIM: Composite Aware Semantic Injection for Text to Motion Generation

**Official Repository**
<!-- **[ICML 2025](https://icml.cc/) | Official Repository** -->


[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://cjerry1243.github.io/casim_t2m/)
[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://arxiv.org/abs/2502.02063)
[![arXiv](https://img.shields.io/badge/arXiv-2502.02063-red)](https://arxiv.org/abs/2502.02063)


This repository provides the official implementation of **CASIM** with 3 SOTA text-to-motion generation models: **MDM, T2MGPT, and CoMo**. All modified code, training scripts, and pretrained checkpoints will be released here.

<!-- ## Abstract 
Recent advances in generative modeling and tokenization have driven significant progress in text-to-motion generation, leading to enhanced quality and realism in generated motions. However, effectively leveraging textual information for conditional motion generation remains an open challenge. We observe that current approaches, primarily relying on fixed-length text embeddings (e.g., CLIP) for global semantic injection, struggle to capture the composite nature of human motion, resulting in suboptimal motion quality and controllability. To address this limitation, we propose the Composite Aware Semantic Injection Mechanism (CASIM), comprising a composite-aware semantic encoder and a text-motion aligner that learns the dynamic correspondence between text and motion tokens. Notably, CASIM is model and representation-agnostic, readily integrating with both autoregressive and diffusion-based methods. Experiments on HumanML3D and KIT benchmarks demonstrate that CASIM consistently improves motion quality, text-motion alignment, and retrieval scores across state-of-the-art methods. Qualitative analyses further highlight the superiority of our composite-aware approach over fixed-length semantic injection, enabling precise motion control from text prompts and stronger generalization to unseen text inputs. -->

---

## 🧠 Overview

**TLDR**. CASIM is a composite-aware semantic injection mechanism for text to motion generation. It is model and representation agnostic. The CASIM-enhanced models, eg. CASIM-MDM, CASIM-T2MGPT, and CASIM-CoMo, show stronger text-motion correspondence, higher motion quality, and better generalizability. 


<p align="left">
  <img src="assets/Teaser.png" alt="CASIM Teaser" width="90%">
</p>

---

## 📦 CASIM Components & Release Status

|                           | Code & Release | Checkpoints | Demo | Evaluation | Training |
|---------------------------|----------------|-------------|------|------------|----------|
| [CASIM-MDM](./CASIM-MDM)  | ✅              | ✅          | ⏳   | ✅          | ✅       |
| CASIM-T2MGPT              | ⏳              | -           | -    | -          |  -       |
| CASIM-CoMo                | ⏳              | -           | -    | -          |  -       |

---


## 📖 Citation

If you find our work useful, please cite with the following bibtex:

```BibTeX
@article{chang2025casim,
  title={CASIM: Composite Aware Semantic Injection for Text to Motion Generation},
  author={Chang, Che-Jui and Liu, Qingze Tony and Zhou, Honglu and Pavlovic, Vladimir and Kapadia, Mubbasir},
  journal={arXiv preprint arXiv:2502.02063},
  year={2025}
}
```

---

## 🙏 Acknowledgement

This repo is built on top of [MDM](https://github.com/GuyTevet/motion-diffusion-model), [T2MGPT](https://github.com/Mael-zys/T2M-GPT), [CoMo](https://github.com/yh2371/CoMo), [T2M](https://github.com/EricGuo5513/text-to-motion), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), and [KIT-ML](https://motion-annotation.humanoids.kit.edu/dataset/). We thank the original authors for their contributions.


---

## 📜 License
This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
See the [LICENSE](LICENSE) file for more details.

