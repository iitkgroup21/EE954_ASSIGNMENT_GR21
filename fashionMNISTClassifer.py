# %% [markdown]
# <a href="https://colab.research.google.com/github/iitkgroup21/EE954_ASSIGNMENT_GR21/blob/anup_e2e_eval/MLP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # About Dataset
# ### Context
# 
# Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.
# 
# The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."
# 
# Zalando seeks to replace the original MNIST dataset
# 
# ### Key Characteristics
# 
# * **Image Resolution:** Each image in the MNIST dataset is 28x28 pixels, with a single color channel (grayscale).
# * **Number of Classes:** The dataset has 10 classes, representing the digits 0 through 9.
# * **Color Format:** Grayscale (1 channel), with pixel values ranging from 0 to 255 in the raw data. After applying transforms.ToTensor(), these values are scaled between 0 and 1.
# 
# ### Dataset Composition
# * **Training Set:** 60,000 images, used for training models.
# * **Test Set:** 10,000 images, used for evaluating model performance.
# 
# ### Typical Usage
# The dataset is often divided into three subsets for practical machine learning workflows:
# 
# * **Training Set (90% of the original training data)**: Used for training the model on 54,000 images.
# * **Validation Set (10% of the original training data):** Used for tuning hyperparameters and preventing overfitting, with 6,000 images.
# * **Test Set (100% of the original Testing data):** Used for final evaluation, with 10,000 images.
# 
# #### Labels
# 
# Each training and test example is assigned to one of the following labels:
# 
# * 0 T-shirt/top
# * 1 Trouser
# * 2 Pullover
# * 3 Dress
# * 4 Coat
# * 5 Sandal
# * 6 Shirt
# * 7 Sneaker
# * 8 Bag
# * 9 Ankle boot
# 
# 
# #### Transformation
# 
# * **ToTensor:** Converts each image to a PyTorch tensor and scales the pixel values to the range [0, 1].
# 

# %% [markdown]
# 
# ## Basic concepts of CNN model :
# 
# A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other.
# 
# Three basic components to define a basic convolutional neural network.
# 
# *   The Convolutional Layer
# *   The Pooling layer
# *   The Output layer
# 
# ![](https://media.licdn.com/dms/image/v2/D5612AQGOui8XZUZJSA/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1680532048475?e=1735776000&v=beta&t=Evq_XWpAo5JDVF4dy5tw2L8E7KDUgYwDrKtnTi5Go_I)
# 
# 
# 
# 

# %%
#pip install wandb

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
wandb.login()


# %%
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch

dataset_location = root = './data/'

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary

# %%
training_dataset = datasets.FashionMNIST(dataset_location,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.FashionMNIST(dataset_location,
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# %%
print("training dataset length =", len(training_dataset))
print("test dataset length =", len(test_dataset))
nc=10 #number of classes

# %%
#concatenate the data so that we can
combined_dataset = torch.utils.data.ConcatDataset([training_dataset, test_dataset])

#Initializing the ratios for the test, training and validation datasets
train_dataset_ratio = 0.9
validation_dataset_ratio = 0.1
test_dataset_ratio = 1.0 # This is applied to the full test data set

#Initalizing the new values of the training, testing and validation data sizes
train_dataset_size = int(train_dataset_ratio * len(training_dataset))
test_dataset_size = int(test_dataset_ratio * len(test_dataset))
validation_dataset_size = int(validation_dataset_ratio * len(training_dataset))

#create the datasets with the sizes

new_train_dataset, new_validation_dataset = torch.utils.data.random_split(training_dataset, [train_dataset_size, validation_dataset_size])
#new_test_dataset = torch.utils.data.random_split(test_dataset, [test_dataset_size]) # This is a redundant step but will be useful if the ratios change

# %%
print(train_dataset_size)
print(test_dataset_size)
print(validation_dataset_size)
print(combined_dataset.cumulative_sizes)
print(len(new_train_dataset))
#print(len(new_test_dataset))
print(len(new_validation_dataset))

# %%
# @title
sample_image, sample_label = new_train_dataset[0]
print("Min pixel value:", sample_image.min().item())
print("Max pixel value:", sample_image.max().item())


# %%
class CustomModel(nn.Module):
     def __init__(self):
       super(CustomModel, self).__init__()
       self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            )
     def forward(self, x):
            x = self.conv_layer(x)
            return x

# %%
# Initialize DataLoaders to retrieve batches of data
train_loader = DataLoader(new_train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(new_validation_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

# %%
#Fully connected layer
class NeuralNetwork:

    def __init__(self):
      self.layers = []
      self.activation_functions = []
      self.layer_data = []
      self.activations = []

    def addnetwork(self, input_size, input_layers_config, output_layers_config):

      """
        - Add the first fully connected layer with the same shape as the flattened CNN output.
        - input_size (int): The size of the input layer.\n",
        - input_layers_config (list): A list of dictionaries specifying the configuration of input layers,
        - output_layers_config (list): A list of dictionaries specifying the configuration of output layers.
        - layers (list): A list to store the layers of the neural network.\n",
        - activation_functions (list): A list to store the activation functions for each layer.\n",

        Args:
        - input_size (int): The size of the input layer.
        - input_layers_config (list): A list of dictionaries specifying the configuration of input layers.
        - output_layers_config (list): A list of dictionaries specifying the configuration of output layers.

        Returns:
        - None

      """
      #addition the first layer
      neurons = input_size  # Same number of neurons as input size
      activation = input_layers_config[0]['activation']
      new_input_layers_config = []

      first_layer_neuron = input_layers_config[0]['neurons']


      # Initialize weights and biases for the first layer
      weights = np.random.randn(input_size, first_layer_neuron)
      bias = np.zeros((1, first_layer_neuron))

      # Store layer information
      self.layer_data.append({
          "Layer": "First Layer",
          "Input Neurons": input_size,
          "Output Neurons": first_layer_neuron,
          "Weights": weights.size,
          "Biases": bias.size,
          "Total Parameters": weights.size + bias.size
      })

      # Append weights, biases, and activation function to the model
      #print("first layer weights =", weights.shape)
      #print("first layer bias =", bias.shape)
      self.layers.append((weights, bias))
      self.activation_functions.append(activation)


      current_input_size = first_layer_neuron

      # Addition of hidden layers
      #Initialize the weights and biases for the input layers
      for i, layer in enumerate(input_layers_config[1:], start=1):
        neurons = layer['neurons']
        activation = layer['activation']

        # Initialize weights and biases for the current layer
        weights = np.random.randn(current_input_size, neurons)   # Weight initialization
        bias = np.zeros((1, neurons))  # Bias initialization

        # Store layer information
        self.layer_data.append({
          "Layer": f"Hidden Layer {i+1}",
          "Input Neurons": current_input_size,
          "Output Neurons": neurons,
          "Weights": weights.size,
          "Biases": bias.size,
          "Total Parameters": weights.size + bias.size
        })

        # Append weights, biases, and activation function to the model
        self.layers.append((weights, bias))
        self.activation_functions.append(activation)

        # Update current input size for next layer
        current_input_size = neurons

      # Addition of output layers
      # Initialize weights and biases for the output layer
      output_neurons = output_layers_config[0]['neurons']
      output_activation = output_layers_config[0]['activation']


      output_weights = np.random.randn(current_input_size, output_neurons)
      output_bias = np.zeros((1, output_neurons))

      self.layer_data.append({
          "Layer": "Output Layer",
          "Input Neurons": current_input_size,
          "Output Neurons": output_neurons,
          "Weights": output_weights.size,
          "Biases": output_bias.size,
          "Total Parameters": output_weights.size + output_bias.size
        })

      # Append output weights, biases, and activation function
      self.layers.append((output_weights, output_bias))
      self.activation_functions.append(output_activation)

    def display_parameters(self):

      """
      Return the parameters of each layer in a DataFrame format.

      Returns:
      - DataFrame: A pandas DataFrame containing parameter details for each layer.
      """
      df = pd.DataFrame(self.layer_data)
      return df

    def flatten(self, X):
      """
      Flatten the input data X.

      Args:
      - X (numpy.ndarray): The input data to be flattened.

      Returns:
      - numpy.ndarray: The flattened data.
      """

      batch_size = X.shape[0]

      #Flatten each image/sample to a 1D vector
      return X.reshape(batch_size, -1) # output(batch size, flattened size)

    # Relu activation function
    def relu(self, x):
        return np.maximum(0, x)

    # Derivative of Relu activation function
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    #Softmax function
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))

        return exps / np.sum(exps, axis=1, keepdims=True)# Softmax activation


    def feedforward(self, X):
      """
      Perform a forward pass through the neural network.

      Args:
      - X (numpy.ndarray): The input to the network from CNN output.

      Returns:
      - yHat (numpy.ndarray): The output of the fully connected network.

      """
      self.activations = [X]  # Initialize with the input
      for i, layer in enumerate(self.layers):

        # Extract weights and biases for the current layer
        weights, bias = layer

        # Debugging shape of X and weights
        #print(f"Layer {i + 1}:")
        #print(f"  Input X shape: {X.shape}")
        #print(f"Weights shape: {weights.shape}")
        #print(f"Bias shape: {bias.shape}")


        # Matrix multiplication and bias addition
        X = np.dot(X, weights) + bias

        # Apply activation function
        if self.activation_functions[i] == 'relu':
          X = self.relu(X)
            #X = np.maximum(0, X)  # Using np.maximum for ReLU activation
        elif self.activation_functions[i] == 'softmax':
          X = self.softmax(X)

        # Store the activation for each layer
        self.activations.append(X)

      yHat = X
      return yHat

    def compute_loss(self, yHat, y):
      # Cross-entropy loss
      m = y.shape[0]
      loss = -np.sum(y * np.log(yHat + 1e-9)) / m
      return loss


    def backwardpass(self, X, y, yHat, learning_rate=0.01):
      """
      print("shape of X")
      print(X.shape)
      print("shape of y")
      print(y.shape)
      print("shape of yHat")
      print(yHat.shape)
      """

      """
      Perform backward propagation through the network and update weights and biases.

      Args:
      - X (numpy.ndarray): The input data.
      - y (numpy.ndarray): The true labels.
      - yHat (numpy.ndarray): The predicted output from the forward pass.
      - learning_rate (float): The learning rate for updating parameters.

      Returns:
      - None
      """
      m = y.shape[0]  # Number of examples in the batch
      # Store the derivatives for each layer
      gradients = []

      # Compute the gradient for the output layer (softmax with cross-entropy loss)
      dA = yHat - y  # Gradient of the loss with respect to output (yHat)



      for i in reversed(range(len(self.layers))):
          #print(f"Backward pass round no: {i}")
          weights, bias = self.layers[i]
          activation = self.activation_functions[i]

          # Use the stored activation as the input to this layer
          A_prev = self.activations[i]

          """
          # Debugging shape of dA and current weights
          print(f"Layer {i + 1}:")
          print(f"  dA shape: {dA.shape}")
          print(f"  weights shape: {weights.shape}")
          print(f"  bias shape: {bias.shape}")
          """
          # Calculate gradients with respect to weights, biases, and inputs for each layer
          if activation == 'softmax':
              dZ = dA  # dZ for softmax layer
          elif activation == 'relu':
              dZ = dA * self.relu_derivative(self.activations[i + 1])




          # Calculate gradients for weights and biases
          dW = np.dot(A_prev.T, dZ) / m
          db = np.sum(dZ, axis=0, keepdims=True) / m


          """
          # Debugging shapes of dW and db
          print(f"Layer {i + 1}:")
          print(f"  A_prev shape: {A_prev.shape}")
          print(f"  dA shape: {dA.shape}")
          print(f"  dZ shape: {dZ.shape}")
          print(f"  weights shape: {weights.shape}")
          print(f"  dW shape: {dW.shape}")
          print(f"  db shape: {db.shape}")

          # Check if shapes align before updating weights
          if dW.shape != weights.shape:
              raise ValueError(f"Shape mismatch: dW shape {dW.shape} does not match weights shape {weights.shape}")
          if db.shape != bias.shape:
              raise ValueError(f"Shape mismatch: db shape {db.shape} does not match bias shape {bias.shape}")
          """

          # Update the weights and biases
          weights -= learning_rate * dW
          bias -= learning_rate * db



          # Update the layer in the network with the new weights and biases
          self.layers[i] = (weights, bias)

          # Update dA for the next layer in the backpropagation process
          dA = np.dot(dZ, weights.T)


    def calculate_accuracy(self, yHat, y):
        # Calculate accuracy based on predictions and true labels
        pred_classes = np.argmax(yHat, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(pred_classes == true_classes)




# %%
input_layers_config = [
    {'neurons': 128, 'activation': 'relu'},
    #{'neurons': 64, 'activation': 'relu'},
    #{'neurons': 32, 'activation': 'relu'}
]
output_layers_config = [
    {'neurons': 10, 'activation': 'softmax'}
]


# %%
# Moving model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
model = CustomModel().to(device)
nn_network = NeuralNetwork()

# Adding layers to the neural network outside the batch loop
cnn_output_sample = next(iter(train_loader))[0]  # Get a sample batch to determine the shape
cnn_output_sample = model(cnn_output_sample.to(device).type(torch.float32))
cnn_output_np_sample = cnn_output_sample.cpu().detach().numpy()
cnn_output_np_sample = nn_network.flatten(cnn_output_np_sample)

# Adding layers to the neural network
nn_network.addnetwork(input_size=cnn_output_np_sample.shape[1], input_layers_config=input_layers_config, output_layers_config=output_layers_config)

wandb.init(
    project="ee954_assignment_gr21",
    group="initial_hyperparameter_training",
    name="lr_0.0001_batch_64_epoch_max100",
    config={
        "learning_rate": 0.0001,
        "epochs": 60,
        "architecture": "CNN+MLP(from scratch)",
        "dataset": "Fashion-MNIST",
        "batch_size": 64,
    }
)

config = wandb.config

# Training Hyperparameters
learning_rate = config.learning_rate
epochs = config.epochs

wandb.watch(model, log="all", log_freq=100)

for epoch in range(epochs):
  print(f"Epoch {epoch + 1}/{epochs}")

  epoch_loss = 0
  correct_predictions = 0
  total_samples = 0
  for i, (x_batch, y_batch) in enumerate( train_loader):

    # Moving input data to the selected device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    y_train = y_batch
    y_train = F.one_hot(y_train, num_classes=nc)

    x_batch = x_batch.type(torch.float32)

    # Perform forward pass with x_batch
    #print("x_batch", x_batch.shape)
    cnn_output = model(x_batch)
    cnn_output_np = cnn_output.cpu().detach().numpy()
    cnn_output_np = nn_network.flatten(cnn_output_np)

    # Feedforward pass
    yhat = nn_network.feedforward(cnn_output_np)

    # Compute Loss (Cross-entropy for classification)

    # Labels are in NumPy format
    y_train = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

    loss = nn_network.compute_loss(yhat, y_train)
    epoch_loss += loss

    # Calculate accuracy
    predicted_labels = np.argmax(yhat, axis=1)
    true_labels = y_batch.cpu().numpy()
    batch_accuracy = (predicted_labels == true_labels).mean()
    correct_predictions += (predicted_labels == true_labels).sum()
    total_samples += y_batch.size(0)

    wandb.log({"batch_loss": loss, "batch accuracy": batch_accuracy, "epoch": epoch})

    # Backpropagation to update weights and bias
    nn_network.backwardpass(cnn_output_np, y_train, yhat, learning_rate)

    # Calculate and log epoch-level metrics to
    epoch_loss /= len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    wandb.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy, "epoch": epoch})



# Retrieve and display the parameter DataFrame
df = nn_network.display_parameters()
print("Parameter Table for Neural Network:")
print(df)

# %%
from sklearn.metrics import confusion_matrix, classification_report
# Set model to evaluation mode
model.eval()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

all_preds = []
all_labels = []

# Disabling gradient calculations for evaluation
with torch.no_grad():
    correct = 0
    total = 0
    for x_test, y_test in test_loader:

        # Move data to the selected device
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Forward pass through CNN
        cnn_output_test = model(x_test.type(torch.float32))

        # Convert to numpy and flatten if needed
        cnn_output_test_np = nn_network.flatten(cnn_output_test.cpu().detach().numpy())


        # Forward pass through fully connected network
        y_pred = nn_network.feedforward(cnn_output_test_np)

        # Calculate accuracy
        predicted = np.argmax(y_pred, axis=1)
        correct += (predicted == y_test.cpu().numpy()).sum()
        total += y_test.size(0)

        all_preds.extend(predicted)
        all_labels.extend(y_test.cpu().numpy())

# Print accuracy
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Calculate and print confusion matrix and classification report
class_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
print("\nClassification Report:")
print(class_report)

pd.DataFrame(confusion_matrix(all_labels, all_preds),index=class_names, columns=class_names)

wandb.finish()


# %%
summary(model, input_size=(1, 28, 28))
print("Output shape:", cnn_output.shape)


