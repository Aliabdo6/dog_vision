# Dog Vision

Dog Vision is a machine learning project that classifies images of dogs into different breeds using a Convolutional Neural Network (CNN). This project leverages TensorFlow, Keras, and Transfer Learning with MobileNetV2 to build and train the model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

The Dog Vision project aims to create a model capable of accurately classifying dog breeds from images. This project demonstrates a multi-class image classification model using TensorFlow and transfer learning.

The uploaded notebook walks through an end-to-end process for building, training, and evaluating the model using TensorFlow and Keras. The project includes data preprocessing, model training, evaluation, and prediction on new images.

### Key Steps:

1. **Data Preparation**:  
   - Loading and preprocessing the dataset.
   - Splitting the data into training, validation, and test sets.

2. **Model Building**:  
   - Utilizing a pre-trained MobileNetV2 from TensorFlow Hub.
   - Adding custom classification layers.

3. **Model Training**:  
   - Training the model with various optimization techniques such as callbacks (TensorBoard and Early Stopping).

4. **Evaluation & Prediction**:  
   - Evaluating model performance on validation data.
   - Generating predictions on test data or custom images.

## Features

- Preprocesses images for training and evaluation.
- Implements Transfer Learning with MobileNetV2.
- Utilizes callbacks for optimizing training (e.g., Early Stopping, TensorBoard).
- Allows saving and loading trained models for future use.
- Provides model predictions on new images.

## Technology Used

![TensorFlow](https://img.shields.io/badge/TensorFlow-%2302569B.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AliAbdo6/dog_vision.git
   cd dog_vision
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Set up the environment:** Install the required libraries as mentioned above.
2. **Prepare the dataset:** Download the dog breed dataset from the Stanford Dogs Dataset and place it in the appropriate directory.
3. **Run the notebook:** Open the `end-to-end-dog-vision.ipynb` notebook, which contains the full pipeline for data preprocessing, training the model, and making predictions.
4. **Customize:** Modify the model, data preprocessing steps, or hyperparameters to experiment with different setups.
   
## Dataset

The project uses the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), which contains images of 120 different dog breeds. You can download the dataset and extract it into the `data/` directory.

## Model Architecture

The architecture used in this project is based on a pre-trained MobileNetV2 for feature extraction, followed by a few dense layers for classification. Transfer learning is applied to leverage the knowledge from MobileNetV2, which was pre-trained on the ImageNet dataset.

Key components:
- **Convolutional base:** Pre-trained MobileNetV2.
- **Custom layers:** Dense layers for classification.
- **Optimizer:** Adam optimizer.
- **Loss function:** Categorical cross-entropy.

## Results

- The model achieves an accuracy of **XX%** on the validation set.
- Detailed performance metrics and visualizations of training history can be found in the `results/` directory or generated from the notebook.
  
## Contributing

Contributions are welcome! Feel free to submit issues or create pull requests for improvements, features, or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [Kaggle Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification)
- [Google Colab](https://colab.research.google.com)

## Contact

Ali Abdo - [LinkedIn](https://www.linkedin.com/in/aliabdo6/)

