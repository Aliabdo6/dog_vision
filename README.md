

# Dog Vision

Dog Vision is a machine learning project that classifies images of dogs into different breeds using a Convolutional Neural Network (CNN). This project leverages TensorFlow and Keras to build and train the model.

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

The Dog Vision project aims to create a model capable of accurately classifying dog breeds from images. This project includes data preprocessing, model training, and evaluation.

This project demonstrates a multi-class image classification model for identifying dog breeds using TensorFlow 2.x and TensorFlow Hub.

The project utilizes a pre-trained MobileNetV2 model from TensorFlow Hub as the base for transfer learning. It involves the following steps:

1. **Data Preparation:**
   - Loading and preprocessing the dog breed dataset.
   - Splitting the data into training and validation sets.
   - Creating data batches for efficient training.

2. **Model Building:**
   - Creating a TensorFlow model using a pre-trained MobileNetV2 layer.
   - Defining the model architecture and optimizer.

3. **Training:**
   - Training the model on a subset of the data to test the workflow.
   - Training the model on the full dataset.
   - Utilizing callbacks like TensorBoard and Early Stopping for monitoring and optimization.

4. **Evaluation:**
   - Evaluating the model's performance on the validation set.
   - Visualizing model predictions and confidence scores.

5. **Saving and Loading:**
   - Saving the trained model for later use.
   - Loading the saved model for prediction.

6. **Prediction:**
   - Making predictions on the test dataset.
   - Preparing the predictions for submission to Kaggle.
   - Making predictions on custom images.

## Features

- Preprocesses dog breed images for model training.
- Implements a CNN using TensorFlow and Keras.
- Evaluates model performance using accuracy and loss metrics.
- Provides visualization for model accuracy and loss over epochs.

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

1. **Set up the environment:** Install the required libraries.
2. **Prepare the dataset:** Download the dog breed dataset and place it in the appropriate directory.
3. **Run the notebook:** Execute the notebook cells to train the model, evaluate its performance, and make predictions.
4. **Customize:** Modify the code to experiment with different models, hyperparameters, and data augmentation techniques.

## Dataset

The dataset used in this project is the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). Download the dataset and extract it to the `data/` directory.

## Model Architecture

The model architecture is based on a Convolutional Neural Network (CNN) with the following layers:

- Convolutional layers
- MaxPooling layers
- Dropout layers
- Dense (Fully connected) layers

## Results

The model achieves an accuracy of XX% on the validation set after YY epochs. Detailed results and training history can be found in the `results/` directory.

## Contributing

Contributions are welcome! Please create a pull request or submit an issue for any bugs, features, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- TensorFlow Hub
- Kaggle Dog Breed Identification competition
- Google Colab

## Contact

Ali Abdo - [LinkedIn](https://www.linkedin.com/in/aliabdo6/)

