# \# ConvNeXt Watermark Detector 🛡️

# 

# \[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat\&logo=pytorch)](https://pytorch.org)

# \[!\[Hardware](https://img.shields.io/badge/Hardware-NVIDIA\_RTX\_4060-76B900.svg?logo=nvidia)](https://www.nvidia.com)

# 

# A robust, high-resolution deep learning pipeline designed to detect complex, semi-transparent, and localized digital watermarks using \*\*ConvNeXt-Tiny\*\* fine-tuned with \*\*Differential Learning Rates\*\* and \*\*Automatic Mixed Precision (AMP)\*\*.

# 

# \---

# 

# \## 📌 Architecture \& Design Highlights

# 

# \- \*\*ConvNeXt Backbone:\*\* Employs modernized inverted bottlenecks with 7x7 depthwise convolutions, delivering a large receptive field tailored for dispersed watermark artifacts.

# \- \*\*High-Resolution Input (384x384):\*\* Maximizes feature fidelity, preventing subtle text and logo edges from downsampling erasure.

# \- \*\*Differential Learning Rates:\*\* Feature extraction stages are adapted gently (lr = 2e-5), while the classification head trains at lr = 2e-4 with AdamW.

# \- \*\*Mixed Precision:\*\* Accelerated with `torch.amp.autocast('cuda')` (FP16), achieving fast convergence on an NVIDIA RTX 4060 Laptop GPU.

# 

# \---

# 

# \## 📊 Benchmark \& Quantitative Results

# 

# Evaluated across validation splits over 7 training epochs:

# 

# | Epoch | Train Loss | Val Loss | Val Accuracy | Precision | Recall | \*\*Val F1-Score\*\* | Status |

# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

# | 1 | 0.4172 | 0.2435 | 91.15% | 90.64% | 91.77% | 0.9120 | Initial Convergence |

# | 2 | 0.2058 | 0.2165 | 92.16% | 92.62% | 91.61% | 0.9212 | Feature Stabilization |

# | 3 | 0.1623 | \*\*0.2143\*\* | 92.62% | 94.20% | 90.84% | 0.9249 | Minimum Val Loss |

# | 4 | 0.1317 | 0.2193 | 93.17% | 94.27% | 91.93% | 0.9308 | Calibration |

# | 5 | 0.1117 | 0.2261 | 93.01% | \*\*94.53%\*\* | 91.30% | 0.9289 | High Precision |

# | 6 | 0.0954 | 0.2291 | \*\*93.40%\*\* | 93.74% | 93.01% | 0.9337 | Balanced Gains |

# | 7 | \*\*0.0887\*\* | 0.2343 | \*\*93.40%\*\* | 93.47% | \*\*0.9332\*\* | \*\*0.9340\*\* | \*\*Optimal Checkpoint\*\* |

# 

# \### 📈 Convergence Dynamics

# !\[Training Dynamics](assets/training\_performance.png)

# 

# \---

# 

# \## 👨‍💻 Author

# \*\*Ammar\*\*  

# \- GitHub: \[@ammar31300](https://github.com/ammar31300)

# \- Field: AI \& Quantum Computing



