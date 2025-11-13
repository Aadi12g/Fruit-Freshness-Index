# 🍎 Fruit Freshness Indicator using CNN, VGG-16 & SSD MobileNet
AI/ML project of a fruit freshness indicator using neural networks.

## 📘 Overview
The Fruit Freshness Indicator is a computer vision-based system that automatically detects and classifies fruits, then predicts their freshness level using deep learning models.  
This project integrates **SSD MobileNet** for object detection and **VGG-16** (along with a custom CNN classifier) for freshness classification.  

The system identifies:
- The **type of fruit**
- The **freshness state** (e.g., Fresh, Rotten, or Stale)
- Performance metrics such as **accuracy, precision, recall, and F1-score**

---

## 🧠 Project Workflow

### 1. **Data Collection**
- Images of multiple fruits were collected and categorized by freshness level.
- Dataset included classes like **Apple, Banana, Orange**, etc.
- Images were preprocessed — resized, normalized, and augmented for better training results.

### 2. **Model Architecture**
#### 🟢 **SSD MobileNet**
- Used for **object detection** — to locate and identify the fruit in the frame.
- Pretrained on the **COCO dataset** for general object detection.
- Fine-tuned for fruit detection.

#### 🔵 **CNN + VGG-16**
- **VGG-16** used for **feature extraction** (transfer learning).
- A **custom CNN classifier** on top predicts the **freshness category**.
- Layers include convolution, ReLU activation, dropout, and dense output layers.

---

## ⚙️ Features
✅ Detects and classifies fruits in an image or live camera feed  
✅ Predicts **fruit freshness** level  
✅ Displays **accuracy, precision, recall, and F1-score** after each run  
✅ Supports both **real-time detection** and **image-based analysis**  
✅ Lightweight and runs on **Linux and Windows**  

---

## 📊 Evaluation Metrics
After every model evaluation, the following metrics are calculated and saved in a CSV file:

| Metric | Description |
|--------|--------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Ratio of true positives to total predicted positives |
| **Recall** | Ratio of true positives to total actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **R1 Score** | Reliability index (custom metric for robustness) |


