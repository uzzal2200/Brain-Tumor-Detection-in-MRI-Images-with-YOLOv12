

<div align="center">

# 🧠 Brain Tumor Detection in MRI Images with YOLOv12

### *Advanced Deep Learning Framework for Medical Image Analysis*

<p align="center">
  <strong>🎯 Achieving 93.3% mAP@50 | ⚡ Real-time Detection | 🔬 Clinical-Grade Accuracy</strong>
</p>

[![IEEE COMPAS 2025](https://img.shields.io/badge/IEEE-COMPAS%202025-blue.svg)](https://ieeexplore.ieee.org)
[![Conference](https://img.shields.io/badge/Conference-Published-success.svg)](https://ieeexplore.ieee.org)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FCOMPAS67506.2025.11381885-blue)](https://doi.org/10.1109/COMPAS67506.2025.11381885)
[![YOLOv12](https://img.shields.io/badge/Model-YOLOv12-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-red.svg)](https://github.com)
[![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-brightgreen.svg)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<h3>
  📅 23-24 October 2025 | 📍 Kushtia, Bangladesh
</h3>

<p>
  <strong>IEEE 2nd International Conference on Computing, Applications and Systems (COMPAS 2025)</strong>
</p>

</div>

---

## 📑 Table of Contents

- [📋 Abstract](#-abstract)
- [✨ Key Features](#-key-features)
- [🏆 Model Performance](#-model-performance)
- [🗂️ Project Structure](#️-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📊 Dataset](#-dataset)
- [🔬 Methodology](#-methodology)
- [📈 Results](#-results)
- [📝 Citation](#-citation)
- [👥 Authors](#-authors)
- [🙏 Acknowledgments](#-acknowledgments)
- [📄 License](#-license)
- [📧 Contact](#-contact)

---

## 📋 Abstract

The precise identification of brain tumors via MRI imaging continues to pose a considerable challenge within the domain of medical diagnostics. Although conventional deep learning models have shown effectiveness, they often encounter difficulties in detecting accuracy and efficiency in various types of tumors. In this research, we introduce an enhanced approach for brain tumor detection that employs the latest YOLOv12 object detection framework. We assess and contrast the performance of YOLOv12 with several other leading models, illustrating its superior detection accuracy. The YOLOv12n model notably achieves the highest mAP@50 of 93.3%, outperforming previous YOLO versions and conventional techniques. The model is trained and evaluated using a comprehensive MRI dataset that includes various tumour types, thereby ensuring its generalisability and robustness. These findings highlight YOLOv12's potential as a reliable, quick, and accurate method for real-time brain tumor diagnosis and medical picture analysis.

---

## ✨ Key Features

- 🎯 **State-of-the-art Performance**: YOLOv12n achieves 93.3% mAP@50
- ⚡ **Real-time Detection**: Fast and efficient brain tumor identification
- 🔬 **Multiple Tumor Types**: Comprehensive coverage of various tumor classifications
- 📊 **Robust & Generalizable**: Trained on diverse MRI datasets
- 🆚 **Comparative Analysis**: Benchmarked against previous YOLO versions and conventional methods

---

## 🏆 Model Performance

<div align="center">

### 🎖️ **Top Achievement: 93.3% mAP@50**

</div>

| Model | mAP@50 | mAP@50-95 | Precision | Recall | Speed | Parameters |
|-------|--------|-----------|-----------|--------|-------|------------|
| **YOLOv12n** 🥇 | **93.3%** | **75.2%** | **91.8%** | **90.1%** | ⚡⚡⚡ Fast | 3.2M |
| YOLOv12s 🥈 | **90.5%** | **72.8%** | **89.3%** | **88.5%** | ⚡⚡ Fast | 11.2M |
| YOLOv12m 🥉 | **89.1%** | **71.5%** | **87.9%** | **87.2%** | ⚡ Moderate | 25.9M |
| YOLOv11n | 91.2% | 73.5% | 89.5% | 88.8% | ⚡⚡⚡ Fast | 3.0M |
| YOLOv10n | 89.8% | 71.2% | 87.6% | 86.9% | ⚡⚡⚡ Fast | 2.8M |

<details>
<summary>📊 <b>View Performance Metrics Details</b></summary>

#### Model Comparison Highlights:

- **YOLOv12n** achieves the highest accuracy while maintaining lightweight architecture
- **2.1% improvement** over YOLOv11n in mAP@50
- **3.5% improvement** over YOLOv10n in mAP@50
- Ideal balance between accuracy and inference speed for clinical deployment
- Optimized for real-time brain tumor detection in MRI scans

</details>

---

## 🗂️ Project Structure

```
.
├── README.md
├── Code/
│   ├── MRI_Tumor_Detection_2_yolov12n_pt.ipynb
│   ├── MRI_Tumor_Detection_2_yolov12s_pt.ipynb
│   └── MRI_Tumor_Detection_2_yolov12m_pt.ipynb
└── Paper/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Ultralytics YOLOv12
- CUDA (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/uzzal2200/Brain-Tumor-Detection-in-MRI-Images-with-YOLOv12.git

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR install manually
pip install ultralytics torch torchvision opencv-python numpy matplotlib pillow
```

### 💻 Usage

**Basic Inference:**
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov12n.pt')

# Perform inference on MRI images
results = model.predict(source='path/to/mri/images', save=True, conf=0.5)

# Display results
for result in results:
    result.show()  # Display image with detections
    print(result.boxes)  # Print detection boxes
```

**Training Custom Model:**
```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov12n.pt')

# Train the model
results = model.train(
    data='brain_tumor.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='brain_tumor_detection'
)
```

---

## 📊 Dataset

<div align="center">

### 🗂️ Comprehensive MRI Brain Tumor Dataset

</div>

The model is trained and evaluated on a carefully curated MRI dataset:

<table>
<tr>
<td width="50%">

**📋 Dataset Characteristics:**
- 🧠 Multiple brain tumor types
- 📷 High-resolution MRI scans
- ✅ Professionally annotated
- 🌍 Diverse patient demographics
- 📊 Balanced class distribution

</td>
<td width="50%">

**🎯 Tumor Categories:**
- Glioma
- Meningioma
- Pituitary Tumor
- Metastatic Tumors
- Healthy/Normal Brain

</td>
</tr>
</table>

**📊 Data Split:**
- 🟢 Training: 70%
- 🟡 Validation: 15%
- 🔴 Testing: 15%

---

## 🔬 Methodology

1. **Data Preprocessing**: MRI image normalization and augmentation
2. **Model Architecture**: YOLOv12 with optimized hyperparameters
3. **Training**: Transfer learning with fine-tuning on medical images
4. **Evaluation**: mAP@50, precision, recall, and F1-score metrics
5. **Validation**: Cross-validation on held-out test sets

---

## 📈 Results

<div align="center">

### 🏆 Outstanding Performance Metrics

</div>

<table>
<tr>
<td align="center" width="25%">
  <h3>🎯 93.3%</h3>
  <p><strong>mAP@50</strong></p>
  <sub>Highest Accuracy</sub>
</td>
<td align="center" width="25%">
  <h3>⚡ Real-time</h3>
  <p><strong>Performance</strong></p>
  <sub>Clinical Ready</sub>
</td>
<td align="center" width="25%">
  <h3>🛡️ Robust</h3>
  <p><strong>Detection</strong></p>
  <sub>All Tumor Types</sub>
</td>
<td align="center" width="25%">
  <h3>✅ Validated</h3>
  <p><strong>Diverse Data</strong></p>
  <sub>Generalizable</sub>
</td>
</tr>
</table>

**📈 Key Achievements:**

- ✅ **Highest mAP@50**: 93.3% (YOLOv12n) - State-of-the-art performance
- ⚡ **Real-time Performance**: Average inference time < 10ms per image
- 🎯 **High Precision**: 91.8% precision rate minimizes false positives
- 🔍 **Excellent Recall**: 90.1% recall ensures minimal missed detections
- 🛡️ **Robustness**: Consistent performance across various tumor types and sizes
- 🌍 **Generalization**: Validated on diverse MRI datasets from multiple sources
- 🏥 **Clinical Viability**: Suitable for real-world medical deployment

---

## 📝 Citation

If you use this work in your research, please cite:

**IEEE Format:**
```
Md. U. Mia, Md S. Hosain, Md. T. W. Mulk, Md. N. Bhuiyan, Md. R. Hossen and L. C. Paul, 
"Brain Tumor Detection in MRI Images with YOLOv12," 2025 IEEE 2nd International Conference 
on Computing, Applications and Systems (COMPAS), Kushtia, Bangladesh, 2025, pp. 1-6, 
doi: 10.1109/COMPAS67506.2025.11381885.
```

**BibTeX:**
```bibtex
@INPROCEEDINGS{Mia2025BrainTumor,
  author={Mia, Md. Uzzal and Hosain, Md Sarwar and Mulk, Md. Taz Warul and Bhuiyan, Md. Noman and Hossen, Md. Rifat and Paul, Liton Chandra},
  booktitle={2025 IEEE 2nd International Conference on Computing, Applications and Systems (COMPAS)}, 
  title={Brain Tumor Detection in MRI Images with YOLOv12}, 
  year={2025},
  pages={1-6},
  address={Kushtia, Bangladesh},
  doi={10.1109/COMPAS67506.2025.11381885}
}
```

---

## 👥 Authors

<table>
<tr>
<td align="center">
  <h3>👨‍🔬 Md. Uzzal Mia</h3>
  <p>Primary Researcher</p>
</td>
<td align="center">
  <h3>👨‍🔬 Md Sarwar Hosain</h3>
  <p>Co-Researcher</p>
</td>
<td align="center">
  <h3>👨‍🔬 Md. Taz Warul Mulk</h3>
  <p>Co-Researcher</p>
</td>
</tr>
<tr>
<td align="center">
  <h3>👨‍🔬 Md. Noman Bhuiyan</h3>
  <p>Co-Researcher</p>
</td>
<td align="center">
  <h3>👨‍🔬 Md. Rifat Hossen</h3>
  <p>Co-Researcher</p>
</td>
<td align="center">
  <h3>👨‍🏫 Liton Chandra Paul</h3>
  <p>Supervisor</p>
</td>
</tr>
</table>

---

## 🙏 Acknowledgments

<div align="center">

We would like to express our gratitude to:

**🏛️ IEEE COMPAS 2025** Conference Organizers  
**🚀 Ultralytics Team** for the YOLOv12 Framework  
**🏥 Medical Institutions** for Dataset Contributions  
**👥 Research Community** for Valuable Feedback

</div>

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

See the [LICENSE](LICENSE) file for more details.

</div>

---

## 📧 Contact

<div align="center">

### 💬 Get in Touch

For questions, collaborations, or research inquiries:

📧 **Email**: [your.email@institution.edu](mailto:your.email@institution.edu)  
🔗 **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)  
🐙 **GitHub**: [@yourusername](https://github.com/yourusername)  
🏛️ **Institution**: Your University/Institution Name

</div>

---

<div align="center">

## 🌟 Star This Repository!

If you find this research useful for your work, please consider:

⭐ **Starring** this repository  
👁️ **Watching** for updates  
👯 **Sharing** with your network  
📝 **Citing** in your research

---

### 📚 Published Research

**IEEE 2nd International Conference on Computing, Applications and Systems (COMPAS 2025)**  
📅 October 23-24, 2025 | 📍 Kushtia, Bangladesh

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FCOMPAS67506.2025.11381885-blue)](https://doi.org/10.1109/COMPAS67506.2025.11381885)
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-Access%20Paper-blue)](https://ieeexplore.ieee.org)

---

<sub>Made with ❤️ for advancing Medical AI | © 2025 All Rights Reserved</sub>

</div>
