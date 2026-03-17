# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

Training data: 60,000 images Test data: 10,000 images Classes: 10 fashion categories The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.

## Neural Network Model
<img width="1082" height="455" alt="image" src="https://github.com/user-attachments/assets/0b99bad5-0532-428f-997f-059e1c6bbb8e" />


## DESIGN STEPS

### STEP 1:Define the Objective
Formulate the task of classifying fashion items (shirts, shoes, bags, etc.) using a CNN model.

### STEP 2:Dataset Preparation
Load the Fashion MNIST dataset and split it into training and testing sets.
### STEP 3: Data Preprocessing

Convert images to tensors

Normalize pixel intensity values

Use DataLoaders for batching and shuffling
### STEP 4:Construct the CNN
Design a neural network with:

Convolutional layers to extract features

ReLU activations for non-linearity

Pooling layers to reduce spatial dimensions

Fully connected layers for final classification
### STEP 5:Train the Model
Use CrossEntropyLoss as the loss function

Optimize with the Adam optimizer

Train over multiple epochs, monitoring loss and accuracy
### STEP 6:Evaluate Performance
Test the trained model on unseen images

Compute accuracy, precision, recall, and F1-score

Generate a confusion matrix to analyze misclassifications
### STEP 7:Deployment and Visualization
Save the trained model for future use

Visualize sample predictions

Integrate the model into applications if required
## PROGRAM

### Name:Deepika R
### Register Number:212224040061
```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
```
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: DEEPIKA R')
        print('Register Number: 212224040061')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```



## OUTPUT
### Training Loss per Epoch
<img width="1065" height="540" alt="image" src="https://github.com/user-attachments/assets/1640bee6-1776-4241-91be-08df0dfe3f93" />
<img width="635" height="209" alt="image" src="https://github.com/user-attachments/assets/c8aa6379-7033-4d26-ac33-80ddc516e356" />

### Confusion Matrix
<img width="934" height="621" alt="image" src="https://github.com/user-attachments/assets/367b14a0-37ed-41ed-b7e9-89771799b619" />

### New Sample Data Prediction
<img width="674" height="443" alt="image" src="https://github.com/user-attachments/assets/9943c270-0c04-4b8c-945e-7b0a7c022c8e" />

### Classification Report

<img width="823" height="500" alt="image" src="https://github.com/user-attachments/assets/84fb287f-4829-489d-a544-9ac43a3c6cf5" />



## RESULT
Thus, a convolutional deep neural network for image classification and to verify the response for new images is to developed successfully
