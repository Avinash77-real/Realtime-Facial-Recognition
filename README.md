# Realtime Facial Details Recognition

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

## 📌 Project Overview
A real-time Python-based application for detecting and classifying human face details across a live web camera feed. This project utilizes Computer Vision and Deep Learning to simultaneously predict **emotion, gender, and age**.

## 🚀 Key Features
* **Real-time Detection:** Uses OpenCV and Haar Cascades for lightweight, rapid face bounding-box detection via webcam.
* **Custom Emotion Recognition:** Features a Convolutional Neural Network (CNN) trained from scratch on the **FER2013 dataset** using TensorFlow/Keras to classify 7 distinct facial expressions.
* **Multi-Model Inference:** Integrates separate Deep Learning models for age and gender prediction on the detected face regions of interest (ROIs).

## 📂 Repository Structure
* `src/train_emotion_model.py`: The CNN architecture and training pipeline for the FER2013 dataset.
* `src/main.py`: The main execution script handling the OpenCV video capture and real-time model inference.
* `models/`: Directory designated for saved `.h5` model weights.

## 🛠️ Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/Realtime-Facial-Recognition.git](https://github.com/Avinash77-real/Realtime-Facial-Recognition.git)
cd Realtime-Facial-Recognition
2.Install dependencies:
pip install -r requirements.txt
3. Run the application
python src/main.py
press 'q' to exit webcam feed
