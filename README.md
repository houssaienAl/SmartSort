🏭 SmartSort: Automated Quality Control with CNNs

![alt text](https://img.shields.io/badge/Python-3.8%2B-blue)


![alt text](https://img.shields.io/badge/TensorFlow-2.x-orange)


![alt text](https://img.shields.io/badge/Deep%20Learning-CNN-red)


![alt text](https://img.shields.io/badge/Status-Completed-success)

SmartSort is a Computer Vision project designed to automate the quality control process in manufacturing environments. Using a Convolutional Neural Network (CNN), this system detects defects in submersible pump impellers with high precision, eliminating the bottlenecks of manual inspection.
📝 The Scenario

TechCast Manufacturing produces metal pump impellers. Historically, the quality control process involved human workers manually inspecting parts on a conveyor belt for 8 hours a day.

    The Problem: Manual inspection is slow, expensive, and prone to errors caused by human fatigue.

    The Objective: Automate the detection of casting defects (cracks, irregularities) using a camera and Deep Learning.

📂 The Dataset

This project uses the Casting Product Image Data for Quality Inspection dataset.

    Source: Kaggle - Ravirajsinh Dabhi

    Content: 6,600+ grayscale images (300x300 pixels).

    Classes:

        ok_front (Defect-free parts)

        def_front (Defective parts with cracks/blowholes)

    Note: Due to file size limits, the dataset is not included in this repository. Please download it from the link above and organize it as described in the Usage section.

🛠️ Tech Stack

    Language: Python

    Framework: TensorFlow / Keras

    Libraries: OpenCV, NumPy, Matplotlib, Pandas

    Environment: Jupyter Notebook / Google Colab

🧠 Model Architecture

We utilize a custom Convolutional Neural Network (CNN) designed for edge detection and texture analysis.

    Input Layer: 128x128 Grayscale images.

    Convolutional Layers: 3 blocks of Conv2D + MaxPooling to extract spatial features (edges, shapes of cracks).

    Flatten Layer: Converts 2D feature maps to 1D vectors.

    Dense Layers: Fully connected layers with ReLU activation.

    Dropout: Applied (0.5) to prevent overfitting.

    Output Layer: Sigmoid activation for binary classification (Probability 0-1).

🚀 Installation & Usage

    Clone the repository
    code Bash

    
git clone https://github.com/YOUR_USERNAME/smart-sort-quality-control.git
cd smart-sort-quality-control

  

Install dependencies
code Bash

    
pip install tensorflow numpy matplotlib opencv-python

  

Setup Data
Download the dataset from Kaggle and structure your folders like this:
code Code

    
Project_Folder/
├── casting_script.py (or .ipynb)
└── dataset/
    └── casting_data/
        ├── train/
        │   ├── def_front/
        │   └── ok_front/
        └── test/
            ├── def_front/
            └── ok_front/

  

Run the Training Script
code Bash

        
    python casting_script.py

      

📊 Results

    Training Accuracy: ~99%

    Validation Accuracy: ~98-99%

    Loss Function: Binary Crossentropy

The model successfully distinguishes between defective and non-defective parts with high confidence, proving that the system is robust enough for real-time deployment on a conveyor belt.

(Optional: Insert a screenshot of your Loss/Accuracy graph here)
🔮 Future Improvements

    Real-time Implementation: Connect the model to a live webcam feed using OpenCV.

    Object Detection: Upgrade to YOLO (You Only Look Once) to localize where the crack is on the part.

    Edge Deployment: Optimize the model using TensorFlow Lite for deployment on Raspberry Pi.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
📜 License

Distributed under the MIT License. See LICENSE for more information.

Created by Houssain Alouani
