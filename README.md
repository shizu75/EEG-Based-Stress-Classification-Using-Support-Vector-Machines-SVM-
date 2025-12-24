# EEG-Based Stress Classification Using Support Vector Machines (SVM)

## Project Overview

This repository presents a **classical machine learning framework for stress detection using EEG signals**, designed and implemented with **research and publication standards** in mind. The study focuses on distinguishing **Stressed vs. Unstressed emotional states** using EEG channel features and a **Support Vector Machine (SVM)** classifier.

The pipeline emphasizes:
- Signal-level exploratory analysis
- Statistical feature inspection
- Class balancing strategies
- Robust model evaluation

This work is suitable for **biomedical signal processing research**, and **affective computing studies**.

---

## Scientific Motivation

Stress recognition from EEG is a critical problem in:
- Mental health monitoring
- Neuroergonomics
- Human–computer interaction
- Wearable brain–computer interfaces (BCI)

Unlike deep learning–heavy approaches, this work demonstrates how **carefully engineered classical ML models** can still achieve strong performance when paired with:
- Thoughtful preprocessing
- Balanced datasets
- Proper statistical validation

---

## Dataset Description

- **Source**: EEG-based emotion dataset
- **EEG Channels Used**:
  - `TP9`
  - `AF7`
  - `AF8`
  - `TP10`
- **Target Variable**: `Emotion`
  - Original labels: `Stressed`, `neutral`, `relaxed`
  - Reformulated labels:
    - `Stressed`
    - `Unstressed` (neutral + relaxed)

### Preprocessing Steps
- Removal of non-informative columns (`timestamps`, `Subject`)
- Label consolidation to reduce ambiguity
- Class balancing through controlled subsampling

---

## Exploratory Data Analysis (EDA)

### Class Distribution
- Histogram visualization used to inspect class imbalance

### Feature Correlation
- Pearson correlation matrix computed
- Diverging heatmaps used to analyze inter-channel relationships

### Channel-Level Distributions
- Kernel density plots for each EEG channel
- Visual inspection of signal variance and distribution overlap

---

## Dataset Balancing Strategy

To prevent classifier bias:
- `Stressed`: first 10,000 samples
- `Unstressed`: symmetric sampling (5,000 from start + 5,000 from end)

This ensures:
- Balanced class representation
- Reduced overfitting
- Stable generalization performance

---

## Feature Encoding & Scaling

- **Label Encoding** applied to target variable
- **StandardScaler** used to normalize EEG feature space
- Essential for distance-based classifiers like SVM

---

## Model Architecture

### Support Vector Machine (SVM)

- **Kernel**: Radial Basis Function (RBF)
- **Hyperparameters**:
  - `C = 1000`
  - `gamma = 0.001`
- **Probability Estimation**: Enabled for log-loss computation

This configuration balances:
- Non-linear decision boundaries
- Generalization capability
- Robustness to noise

---

## Model Evaluation

### Metrics Used
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- Log Loss (probabilistic confidence)

### Visualization
- Confusion matrix heatmap
- Predicted class probabilities
- Validation loss reporting

These metrics provide both **classification performance** and **uncertainty awareness**, critical for biomedical applications.

---

## Key Results

- High classification accuracy on balanced EEG data
- Clear separation between stressed and unstressed states
- Stable probabilistic predictions (low log loss)
- Interpretable classical ML pipeline

---

## Research & PhD Relevance

This project demonstrates:

- EEG signal understanding beyond black-box models
- Responsible class balancing for affective datasets
- Statistical EDA before model fitting
- Classical ML rigor aligned with neuroscience research norms

It serves as:
- A **strong applied neuroscience ML project**
- A baseline for hybrid ML–DL EEG systems
- A reference implementation for stress detection research

---

## Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Support Vector Machines (SVM)

---

## Ethical & Academic Use

This work is intended strictly for:
- Academic research
- Educational purposes
- Non-clinical decision support

Clinical deployment would require extensive validation and ethical approval.

---

## Author Note

Developed as part of a **neuroengineering and AI research portfolio**, emphasizing **signal processing, interpretability, and reproducibility**, aligned with **PhD-level research standards**.
