# Skin Disease Detection Using Deep Learning

## Overview
This project focuses on the detection of skin diseases using deep learning techniques, specifically leveraging Convolutional Neural Networks (CNNs). The primary objective is to develop an automated system capable of identifying various skin conditions based on dermatoscopic images. By utilizing transfer learning and data augmentation techniques, the model aims to achieve high accuracy and efficiency in diagnosing skin diseases.

## Dataset
The project utilizes the **HAM10000** dataset, a comprehensive collection of dermatoscopic images containing various skin lesions. The dataset includes images of multiple skin conditions, making it an ideal resource for training a robust skin disease detection model.

**Key Features of HAM10000 Dataset:**
- Contains **10,015** high-resolution dermatoscopic images
- Covers **seven** common types of skin lesions, including melanoma, benign keratosis, and dermatofibroma
- Well-annotated and widely used for research in dermatology and medical AI

## Models Implemented

### 1. Sequential API with Conv2D
- Employs a **sequential** architecture using **Conv2D** layers for feature extraction.
- Designed as a **baseline model** to compare performance with more advanced architectures.
- Achieves an **accuracy of 68%** on the validation set.
- Suitable for understanding the impact of convolutional layers on skin disease classification.

### 2. MobileNet-V2
- Utilizes the **MobileNet-V2** architecture, known for its efficiency and lightweight design, making it suitable for **real-time applications**.
- Implements **transfer learning**, leveraging pre-trained weights to enhance accuracy.
- Achieves an **accuracy of 82%** on the validation set, demonstrating a significant improvement over the baseline model.
- Recommended for practical skin disease detection due to its balance between accuracy and computational efficiency.

## Training Process
The models were trained using **data augmentation** and **transfer learning** to enhance generalization and improve performance. The key training steps include:
1. **Preprocessing:** Image resizing, normalization, and augmentation (flipping, rotation, zooming, etc.).
2. **Model Training:** Fine-tuning the MobileNet-V2 model with pre-trained ImageNet weights.
3. **Evaluation:** Performance assessment using validation and test datasets.
4. **Optimization:** Using techniques like **Adam optimizer, categorical cross-entropy loss**, and **learning rate scheduling**.

## How to Use

### Requirements
Ensure that the necessary libraries and dependencies are installed. You can install them using:
```sh
pip install -r requirements.txt
```

### Dataset Preparation
1. Download the **HAM10000 dataset** from the official source or Kaggle.
2. Organize the dataset into the required directory structure.
3. Update the dataset path in the configuration file or scripts if needed.

### Training the Model
To train the model, run the following command:
```sh
python train.py --model mobilenet
```
or for the Sequential model:
```sh
python train.py --model sequential
```

### Evaluating the Model
After training, evaluate the model on a test set:
```sh
python evaluate.py --model_path path/to/saved/model
```

### Making Predictions
To classify a new image, run the prediction script:
```sh
python predict.py --image path/to/image.jpg --model_path path/to/saved/model
```

## Results
- **MobileNet-V2** outperforms the Sequential API model, achieving an accuracy of **82%**.
- The model successfully identifies different types of skin lesions with high precision.
- Transfer learning proves to be an effective approach for medical image classification.

## Future Enhancements
- Integration of additional datasets to improve generalization.
- Deployment of the model as a **web or mobile application** for real-world usability.
- Implementing **explainability techniques** (e.g., Grad-CAM) to visualize model decisions.
- Fine-tuning with more advanced architectures like **EfficientNet or Vision Transformers**.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## Contact
For any questions or feedback, please reach out via GitHub issues or email.


