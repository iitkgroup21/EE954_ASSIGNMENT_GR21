# MLP Fashion-MNIST Classification

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Results](#results)
- [Usage Instructions](#usage-instructions)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project involves building a Convolutional Neural Network (CNN) to classify images from the Fashion-MNIST dataset using PyTorch. The goal is to provide a benchmark for deep learning algorithms on fashion-related items.

## Dataset
### Context
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

### Key Characteristics
- **Image Resolution:** 28x28 pixels, single-channel (grayscale)
- **Number of Classes:** 10
- **Color Format:** Grayscale, pixel values range from 0 to 255, scaled to [0, 1] using transforms.ToTensor()

### Dataset Composition
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images

### Labels
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## Model Architecture
### CNN Model
The CNN model consists of the following layers:
- **Convolutional Layer:** 3 layers with increasing depth (16, 32, 64 filters)
- **Batch Normalization:** Applied after each convolutional layer
- **ReLU Activation:** Applied after each batch normalization
- **Max Pooling:** Applied after every convolutional layer

### Fully Connected Layer
The fully connected network is defined to process the output from the CNN and produce the final classification.

## Training Procedure
The training procedure involves:
1. **Data Preparation:** Download and preprocess the Fashion-MNIST dataset.
2. **Model Initialization:** Define the CNN and fully connected network.
3. **Training Loop:** Train the model using the training set, validate using the validation set, and evaluate using the test set.
4. **Logging:** Use Weights & Biases (wandb) for experiment tracking.

## Results
The performance of the model is evaluated in terms of accuracy on the test set.

## Usage Instructions
### Installation
To run this project, you need to have Python and the following packages installed:

```bash
pip install numpy matplotlib pandas torch torchvision wandb
```

### Running the Code
1. Clone this repository:
```bash
git clone https://github.com/iitkgroup21/EE954_ASSIGNMENT_GR21.git
cd EE954_ASSIGNMENT_GR21
```
2. Run the Jupyter notebook:
```bash
jupyter notebook MLP.ipynb
```

## Dependencies
- Python
- NumPy
- Matplotlib
- Pandas
- PyTorch
- Torchvision
- Weights & Biases

To run the notebook via Google Colab without installing anything locally, follow these steps:

1. Open the notebook in GitHub: [MLP.ipynb](https://github.com/iitkgroup21/EE954_ASSIGNMENT_GR21/blob/main/MLP.ipynb)
2. Click on the "Open in Colab" badge at the top of the notebook: 

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iitkgroup21/EE954_ASSIGNMENT_GR21/blob/main/MLP.ipynb)

3. This will open the notebook in Google Colab, where you can run all the cells in the notebook without needing to install any dependencies locally.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
