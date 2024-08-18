from imageio import imread
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import models

training_data_path = "../data/training_data"
training_data_classifications_path = "../data/training_classifications.csv"

validation_data_path = "../data/validation_data"
validation_data_classifications_path = "../data/validation_classifications.csv"

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


num_training_images = 10000
num_validation_images = 1000

num_channels = 3
image_height = 225
image_width = 400


############################################################################
################################### DATA ###################################
############################################################################


column_names = ['filename', 'label0', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9']


# def get_sampler(dataset, num_samples):
#     indices = np.random.choice(len(dataset), num_samples, replace=False)
#     return SubsetRandomSampler(indices)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, names=column_names)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = imread(img_path, format="ppm")
        image = (image / 255).astype('float32')
        image = torch.from_numpy(image).to(device)
        image = image.permute(2, 0, 1)

        labels = self.img_labels.iloc[idx, 1:].values
        labels = labels.astype(bool)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)

        return image, labels


training_data = CustomImageDataset(
    training_data_classifications_path,
    training_data_path,
)

validation_data = CustomImageDataset(
    validation_data_classifications_path,
    validation_data_path,
)


############################################################################
############################## NEURAL NETWORK ##############################
############################################################################


network = "VGG-16"
cost_function = "binary cross-entropy"

num_output_neurons = len(labels_map)
mini_batch_size = 25
num_epochs = 10000
learning_rate = 0.005

regularizer = "L2"
regularization_parameter = 0.1 / mini_batch_size


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(image_height * image_width, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, num_output_neurons),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits
    

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

binary_cross_entropy_cost = nn.BCELoss()

# Use a basic stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization_parameter if regularizer == "L2" else 0)

# To implement a variable learning rate schedule
lowest_cost_so_far = float('inf')
batches_since_improvement = 0
learning_rate_decrease_threshold = 20

with open("../data/log.txt", 'w') as file:
    # Train for some number of epochs
    for epoch in range(num_epochs): 
        # sampler = get_sampler(training_data, num_training_images)
        # training_dataloader = DataLoader(training_data, batch_size=mini_batch_size, sampler=sampler)
        training_dataloader = DataLoader(training_data, batch_size=mini_batch_size, shuffle=True)
        mini_batch_number = 0
        for mini_batch, mini_batch_classifications in training_dataloader:
            print(f"epoch: {epoch} | mini_batch_number: {mini_batch_number}")
            
            # A 2-dimensional output. Each row is the predicted classification for a particular training image in the mini-batch
            training_prediction = model(mini_batch)

            # Define the cost function
            if cost_function == "quadratic":
                cost = (((mini_batch_classifications - training_prediction) ** 2) / (2 * mini_batch_size)).sum()
            elif cost_function == "binary cross-entropy":
                cost = binary_cross_entropy_cost(training_prediction, mini_batch_classifications)

            # Store the lowest cost so far in order to determine, after some number of batches of no improvement, when to decrease the learning rate while training
            if cost < lowest_cost_so_far:
                lowest_cost_so_far = cost 
                batches_since_improvement = 0
            else:
                batches_since_improvement += 1

            # Halve the learning rate when our cost hasn't decreased beyond what we've seen so far in some number of mini-batches
            if batches_since_improvement >= learning_rate_decrease_threshold:
                learning_rate /= 2
                for g in optimizer.param_groups:
                    g["lr"] = learning_rate
                
                lowest_cost_so_far = float('inf')
                batches_since_improvement = 0

            print(f"Cost: {cost}")

            # Backpropagation
            cost.backward()

            # Update weights and biases 
            optimizer.step()

            # Zero the gradients so they don't accumulate 
            optimizer.zero_grad(set_to_none=False)

            mini_batch_number += 1

        # Check the classifier accuracy against the validation data every epoch
        # sampler = get_sampler(validation_data, num_validation_images)
        # validation_dataloader = DataLoader(validation_data, batch_size=1, sampler=sampler)
        validation_dataloader = DataLoader(validation_data, batch_size=1)
        correct = 0
        for sample_validation_image, sample_validation_classification in validation_dataloader:
            # A 1-D tensor with size (num_output_neurons,)
            validation_prediction = model(sample_validation_image).squeeze()
            
            # A 1-D tensor with size (num_output_neurons,)
            sample_validation_classification = sample_validation_classification.squeeze()

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
