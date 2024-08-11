from imageio import imread
import matplotlib.pyplot as plt
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from process_data import training_images, validation_images, training_image_classifications, validation_image_classifications, training_data_path, validation_data_path

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

num_training_images = 3000
num_validation_images = 300

image_height = 225
image_width = 400

num_input_neurons = image_height * image_width
num_output_neurons = len(labels_map)
mini_batch_size = 10
num_epochs = 30
learning_rate = 0.1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(image_height * image_width, 500),
            nn.ReLU(),
            nn.Linear(500, num_output_neurons),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

# Use a basic stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

with open("../data/log.txt", 'w') as file:
    # Train for some number of epochs
    for epoch in range(num_epochs):
        for batch_start_index in range(0, num_training_images, mini_batch_size):
            print(f"epoch: {epoch} | batch_start_index: {batch_start_index}")
            
            mini_batch = torch.empty(0, image_height, image_width, dtype=torch.float32).to(device)
            mini_batch_classifications = torch.empty(0, num_output_neurons).to(device)
            for training_image_index in range(batch_start_index, batch_start_index + mini_batch_size):
                training_image_name = training_images[training_image_index]
                training_image_path = os.path.join(training_data_path, training_image_name)

                sample_training_image = imread(training_image_path, format="ppm")
                sample_training_image = sample_training_image[:,:,0]
                sample_training_image = (sample_training_image / 255).astype('float32')
                sample_training_image = torch.from_numpy(sample_training_image).unsqueeze(0).to(device)
                mini_batch = torch.cat((mini_batch, sample_training_image), dim=0)

                mini_batch_classification = training_image_classifications[training_image_name]
                mini_batch_classification = torch.tensor(mini_batch_classification).unsqueeze(0).to(device)
                mini_batch_classifications = torch.cat((mini_batch_classifications, mini_batch_classification), dim=0)

            mini_batch.requires_grad_(False)

            # A 2-dimensional output. Each row is the predicted classification for a particular training image in the mini-batch
            training_prediction = model(mini_batch)

            # Define the quadratic cost function
            quadratic_cost = (((mini_batch_classifications - training_prediction) ** 2) / (2 * mini_batch_size)).sum()
            print(f"Quadratic cost: {quadratic_cost}")

            # Backpropagation
            quadratic_cost.backward()

            # Update weights and biases 
            optimizer.step()

            # Zero the gradients so they don't accumulate 
            optimizer.zero_grad(set_to_none=False)
        
        # Check the classifier accuracy against the validation data every epoch
        correct = 0
        for validation_image_index in range(num_validation_images):
            validation_image_name = validation_images[validation_image_index]
            validation_image_path = os.path.join(validation_data_path, validation_image_name)

            sample_validation_image = imread(validation_image_path, format="ppm")
            sample_validation_image = sample_validation_image[:,:,0]
            sample_validation_image = (sample_validation_image / 255).astype('float32')
            # We have a batch size of 1 in this case. We want to add an extra dimension to the tensor, 
            # like this: (H, W) -> (1, H, W) - hence .unsqueeze(0)
            sample_validation_image = torch.from_numpy(sample_validation_image).unsqueeze(0).to(device)

            sample_validation_classification = validation_image_classifications[validation_image_name]
            sample_validation_classification = torch.tensor(sample_validation_classification).to(device)

            # A 2-D tensor with size (1, num_output_neurons)
            validation_prediction = model(sample_validation_image)

            threshold = 0.5 
            difference = torch.abs(sample_validation_classification - validation_prediction)
            result = torch.all(difference < threshold).item()  # Returns a bool
            correct += int(result)

        validation_accuracy = correct / num_validation_images
        print(f"Validation accuracy after epoch {epoch}: {validation_accuracy} ")

        file.write(f"{epoch}: {validation_accuracy:.4f} {quadratic_cost}\n")
        file.flush()
