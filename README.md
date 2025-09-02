# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.)

### STEP 2: 
Download and preprocess the MNIST dataset using transforms.

### STEP 3: 
Create a CNN model with convolution, pooling, and fully connected layers.

### STEP 4: 
Set the loss function and optimizer. Move the model to GPU if available.

### STEP 5: 
Train the model using the training dataset for multiple epochs.

### STEP 6: 
Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).

## PROGRAM

### Name:Saranya V

### Register Number:212223040188

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        #Include your code here

    def forward(self, x):
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer
model =
criterion =
optimizer =

def train_model(model, train_loadr, num_epochs=10):
    #Include your code here

```

### OUTPUT

## Training Loss per Epoch
<img width="252" height="220" alt="dl1" src="https://github.com/user-attachments/assets/0944cd1e-2d06-4099-afb6-3a49bd2f3c3e" />

## Confusion Matrix
<img width="532" height="460" alt="dl2" src="https://github.com/user-attachments/assets/596843ac-8725-45e0-9df3-abfd1c176b76" />

## Classification Report
<img width="410" height="337" alt="dl3" src="https://github.com/user-attachments/assets/fe1d5fdd-76c1-4ce2-898c-f3d184bb52f1" />

### New Sample Data Prediction
<img width="390" height="452" alt="dl4" src="https://github.com/user-attachments/assets/d3503330-2d80-48b3-9a42-ae4f7b3a329e" />

## RESULT
Developing a convolutional neural network (CNN) classification model for the given dataset was executed successfully.
