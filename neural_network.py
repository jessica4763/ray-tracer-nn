from imageio import imread
import matplotlib.pyplot as plt
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from process_data import training_image_names, validation_image_names, training_classifications, validation_classifications, training_data_path, validation_data_path

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

labels_map = {
    0: "Did not normalise point to light vector.",
    1: "Did not normalise view vector.",
    2: "Did not apply the max function in the diffuse calculation.",
    3: "Did not apply the max function in the specular calculation.",
    4: "Applied the pow function inside the max function.",
    5: "Did not negate the point to light normalised vector inside reflect.",
    6: "Calculated a light to point vector instead of a point to light vector.",
    7: "Calculated a camera to point vector instead of a point to camera vector.",
    8: "Did not consider light colour in diffuse calcluation.",
    9: "Did not nudge shadow ray.",
}

num_training_images = 1000
num_validation_images = 100

image_height = 225
image_width = 400

network_type = "CNN"
cost_function = "quadratic"

num_input_neurons = image_height * image_width
num_output_neurons = len(labels_map)
mini_batch_size = 10
num_epochs = 100
learning_rate = 0.01

regularizer = "L2"
regularization_parameter = 1 / mini_batch_size


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(image_height * image_width, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, num_output_neurons),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits


class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 6 x 6 square convolution kernel
        # Apply 5 x 5 max-pooling after the convolutional layer
        self.conv1 = torch.nn.Conv2d(1, 6, 6)
        # 6 input image channels, 16 output channels, 5 x 5 square convolutional layer
        # Apply 5 x 5 max-pooling after the convolutional layer
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 15 * 8, 120)  # 15 * 8 from resulting image dimension after both convolutional and max-pooling layers
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (5, 5) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (5, 5))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 5)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

# Load images and their classifications into memory 
def load_images_and_classifications(data_path, image_names, num_images, image_classifications):
    loaded_images = torch.empty(0, image_height, image_width, dtype=torch.float32).to(device)
    loaded_classifications = torch.empty(0, num_output_neurons).to(device)

    for image_index in range(num_images):
        image_name = image_names[image_index]
        image_path = os.path.join(data_path, image_name)

        sample_image = imread(image_path, format="ppm")
        sample_image = sample_image[:,:,0]
        sample_image = (sample_image / 255).astype('float32')
        sample_image = torch.from_numpy(sample_image).unsqueeze(0).to(device)
        loaded_images = torch.cat((loaded_images, sample_image), dim=0)

        sample_classification = image_classifications[image_name]
        sample_classification = torch.tensor(sample_classification).unsqueeze(0).to(device)
        loaded_classifications = torch.cat((loaded_classifications, sample_classification), dim=0)

        if image_index % 10 == 0:
            print(f"{image_index} images loaded")

    return loaded_images, loaded_classifications


loaded_training_images, loaded_training_classifications = load_images_and_classifications(training_data_path, training_image_names, num_training_images, training_classifications)
loaded_validation_images, loaded_validation_classifications = load_images_and_classifications(validation_data_path, validation_image_names, num_validation_images, validation_classifications)
model = ConvolutionalNeuralNetwork().to(device) if network_type == "CNN" else NeuralNetwork().to(device)

# Define cross-entropy cost function
binary_cross_entropy_cost = nn.BCELoss()

# Use a basic stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization_parameter if regularizer == "L2" else 0)

with open("../data/log.txt", 'w') as file:
    # Train for some number of epochs
    for epoch in range(num_epochs):
        for batch_start_index in range(0, num_training_images, mini_batch_size):
            print(f"epoch: {epoch} | batch_start_index: {batch_start_index}")
            
            mini_batch = loaded_training_images[batch_start_index:batch_start_index + mini_batch_size, :, :].unsqueeze(1)  # .unsqueeze(1) when using the CNN
            mini_batch_classifications = loaded_training_classifications[batch_start_index:batch_start_index + mini_batch_size, :]
            mini_batch.requires_grad_(False)
            
            # A 2-dimensional output. Each row is the predicted classification for a particular training image in the mini-batch
            training_prediction = model(mini_batch)

            # Define the cost function
            if cost_function == "quadratic":
                cost = (((mini_batch_classifications - training_prediction) ** 2) / (2 * mini_batch_size)).sum()
            elif cost_function == "binary cross entropy":
                cost = binary_cross_entropy_cost(training_prediction, mini_batch_classifications)

            print(f"Cost: {cost}")

            # Backpropagation
            cost.backward()

            # Update weights and biases 
            optimizer.step()

            # Zero the gradients so they don't accumulate 
            optimizer.zero_grad(set_to_none=False)
        
        # Check the classifier accuracy against the validation data every epoch
        correct = 0
        for validation_image_index in range(num_validation_images):
            sample_validation_image = loaded_validation_images[validation_image_index, :, :].unsqueeze(0).unsqueeze(0) # .unsqueeze(0).unsqueeze(0) when using the CNN
            sample_validation_classification = loaded_validation_classifications[validation_image_index, :]

            # A 2-D tensor with size (1, num_output_neurons)
            validation_prediction = model(sample_validation_image)

            # The prediction is correct if all predicted labels are within 0.5 of the actual labels 
            difference = torch.abs(sample_validation_classification - validation_prediction)
            result = torch.all(difference < 0.5).item()  # Returns a bool
            correct += int(result)

        validation_accuracy = correct / num_validation_images
        print(f"Validation accuracy after epoch {epoch}: {validation_accuracy} ")

        file.write(f"{epoch}: {validation_accuracy:.4f} {cost}\n")
        file.flush()
