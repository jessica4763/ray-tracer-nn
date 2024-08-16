from imageio import imread
import matplotlib.pyplot as plt
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
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

num_channels = 3
image_height = 225
image_width = 400

network = "VGG-16"
cost_function = "binary cross-entropy"

num_output_neurons = len(labels_map)
mini_batch_size = 10
num_epochs = 1000
learning_rate = 0.01

regularizer = "L2"
regularization_parameter = 0.1 / mini_batch_size


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
    

# Load images and their classifications into memory 
def load_images_and_classifications(data_path, image_names, num_images, image_classifications):
    loaded_images = torch.empty(0, num_channels, image_height, image_width, dtype=torch.float32).to(device)
    loaded_classifications = torch.empty(0, num_output_neurons).to(device)

    for image_index in range(num_images):
        image_name = image_names[image_index]
        image_path = os.path.join(data_path, image_name)

        sample_image = imread(image_path, format="ppm")
        # sample_image = sample_image[:, :, 0]
        sample_image = (sample_image / 255).astype('float32')
        sample_image = torch.from_numpy(sample_image).unsqueeze(0).to(device)
        sample_image = sample_image.permute(0, 3, 1, 2)
        loaded_images = torch.cat((loaded_images, sample_image), dim=0)

        sample_classification = image_classifications[image_name]
        sample_classification = torch.tensor(sample_classification).unsqueeze(0).to(device)
        loaded_classifications = torch.cat((loaded_classifications, sample_classification), dim=0)

        if image_index % 10 == 0:
            print(f"{image_index} images loaded")

    return loaded_images, loaded_classifications


loaded_training_images, loaded_training_classifications = load_images_and_classifications(training_data_path, training_image_names, num_training_images, training_classifications)
loaded_validation_images, loaded_validation_classifications = load_images_and_classifications(validation_data_path, validation_image_names, num_validation_images, validation_classifications)

if network == "VGG-16":
    model = models.vgg16(pretrained=True).to(device)
    new_classifier = nn.Sequential(
        *list(model.classifier.children())[:6],  # Keep the first 6 layers of the original classifier. Note that the new classifier uses references to the layers of the original classifier. 
        nn.Linear(model.classifier[6].in_features, num_output_neurons),  # Final output layer
        nn.Sigmoid()
    )
    model.classifier = new_classifier.to(device)
else:
    model = NeuralNetwork().to(device)

# Define binary cross-entropy cost function
binary_cross_entropy_cost = nn.BCELoss()

# Use a basic stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization_parameter if regularizer == "L2" else 0)

with open("../data/log.txt", 'w') as file:
    # Train for some number of epochs
    for epoch in range(num_epochs):
        for batch_start_index in range(0, num_training_images, mini_batch_size):
            print(f"epoch: {epoch} | batch_start_index: {batch_start_index}")
            
            mini_batch = loaded_training_images[batch_start_index:batch_start_index + mini_batch_size, :, :]
            mini_batch_classifications = loaded_training_classifications[batch_start_index:batch_start_index + mini_batch_size, :]
            mini_batch.requires_grad_(False)
            
            # A 2-dimensional output. Each row is the predicted classification for a particular training image in the mini-batch
            training_prediction = model(mini_batch)

            # Define the cost function
            if cost_function == "quadratic":
                cost = (((mini_batch_classifications - training_prediction) ** 2) / (2 * mini_batch_size)).sum()
            elif cost_function == "binary cross-entropy":
                cost = binary_cross_entropy_cost(training_prediction, mini_batch_classifications)

            print(f"Cost: {cost}")

            # Backpropagation
            cost.backward()

            # Update weights and biases 
            optimizer.step()

            # Zero the gradients so they don't accumulate 
            optimizer.zero_grad(set_to_none=False)

            # Decrease the learning rate if the validation accuracy doesn't increase for 
        
        # Check the classifier accuracy against the validation data every epoch
        correct = 0
        for validation_image_index in range(num_validation_images):
            sample_validation_image = loaded_validation_images[validation_image_index, :, :].unsqueeze(0)
            sample_validation_classification = loaded_validation_classifications[validation_image_index, :]

            # A 2-D tensor with size (1, num_output_neurons)
            validation_prediction = model(sample_validation_image).squeeze()

            # # The prediction is correct if all predicted labels are within 0.5 of the actual labels 
            # difference = torch.abs(sample_validation_classification - validation_prediction)
            # result = torch.all(difference < 0.5).item()  # Returns a bool
            # correct += int(result)

            # The prediction is correct if all predicted labels are within 0.5 of the actual labels 
            difference = torch.abs(sample_validation_classification - validation_prediction)

            # Create a boolean tensor where each element is True if the difference is less than 0.5
            labels_correct = difference < 0.5

            # Count the number of True values in the boolean tensor
            num_labels_correct = torch.sum(labels_correct).item()

            # Determine the total number of elements in the boolean tensor
            total_labels = labels_correct.numel()

            # Result is True if all or all but one of the elements are True
            result = (num_labels_correct == total_labels) or (num_labels_correct == (total_labels - 1)) # or (num_labels_correct == (total_labels - 2))

            # result = torch.all(difference < 0.5).item()  # Returns a bool
            correct += int(result)

        validation_accuracy = correct / num_validation_images
        print(f"Validation accuracy after epoch {epoch}: {validation_accuracy} ")

        file.write(f"{epoch}: {validation_accuracy:.4f} {cost}\n")
        file.flush()
