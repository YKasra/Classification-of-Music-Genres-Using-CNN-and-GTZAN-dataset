# 🎵 Music Genre Classification Using CNN

A Convolutional Neural Network that automatically classifies music into 10 genres from audio data, achieving **94.89% test accuracy**.

> **Bachelor's Final Project** — Azad University of Tehran (Grade A)

---

## Overview

This project tackles the challenge of automated music genre identification using deep learning. The system extracts audio features from raw music files and feeds them into a CNN architecture for multi-class classification across 10 genres.

### Results at a Glance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 94.89% |
| **Precision** | 0.952 |
| **Recall** | 0.946 |
| **F1 Score** | 0.940 |
| **Training Accuracy** | 99.27% |

---

## Dataset

**GTZAN Genre Collection** — the most widely used benchmark dataset for music genre recognition.

- **1,000 audio samples** (30 seconds each)
- **10 genres:** Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
- **100 tracks per genre**
- Collected from diverse sources (CDs, radio, microphone recordings) to represent varied recording conditions

---

## Methodology

### Feature Extraction

Audio features were extracted using the **Librosa** library:

- **Spectrogram Visualization** — Time-frequency representation of audio signals
- **Spectral Rolloff** — Frequency below which a specified percentage of total spectral energy lies
- **Chroma Features** — Representation of the 12 pitch classes
- **Zero-Crossing Rate** — Rate at which the signal changes sign

All features were scaled using a standard scaler before model training.

### Model Architecture

- **Type:** Convolutional Neural Network (CNN)
- **Layers:** Multiple dense layers with ReLU activation
- **Regularization:** Dropout layers to prevent overfitting
- **Optimizer:** Adam
- **Loss Function:** Sparse categorical cross-entropy
- **Epochs:** 550

### Prediction Example

The model takes an audio file as input and outputs genre probability distribution:

[url=https://freeimage.host/i/qKUbqhB][img]https://iili.io/qKUbqhB.th.jpg[/img][/url]

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **TensorFlow / Keras** | CNN model building and training |
| **Librosa** | Audio feature extraction |
| **Scikit-learn** | Data preprocessing, train/test split |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib** | Visualization |

---

## Project Structure

```
├── data/                    # GTZAN dataset (not included — see below)
├── model/                   # Saved trained model
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── predict.py
├── results/                 # Output charts and metrics
└── README.md
```

### Getting the Dataset

Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) or [the original source](http://marsyas.info/downloads/datasets.html).

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/ykasra/Classification-of-Music-Genres-Using-CNN-and-GTZAN-dataset.git
cd Classification-of-Music-Genres-Using-CNN-and-GTZAN-dataset

# Install dependencies
pip install tensorflow librosa scikit-learn pandas numpy matplotlib

# Train the model
python src/train_model.py

# Predict genre of an audio file
python src/predict.py --input path/to/audio.wav
```

---

## Future Work

- Experiment with larger, more diverse music datasets
- Explore alternative architectures (LSTM, Transformer-based models)
- Add real-time audio classification via microphone input
- Deploy as a web application with Streamlit

---

## References

1. Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). *"Convolutional recurrent neural networks for music classification."*
2. Tzanetakis, G., & Cook, P. (2002). *"Musical genre classification of audio signals."*
3. Goodfellow, I., Bengio, Y., & Courville, A. — *Deep Learning*
4. Bishop, C. M. — *Pattern Recognition and Machine Learning*

---

## Author

**Kasra Yaraei** — [LinkedIn](https://linkedin.com/in/kasrayaraei) · [Email](mailto:Kasrayaraei@gmail.com)
