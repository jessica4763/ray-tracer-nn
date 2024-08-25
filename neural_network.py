from imageio import imread
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import models

print(torch.__version__)
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())

# training_data_path = "../data/simple_training_data_05679"
# training_data_classifications_path = "../data/simple_training_classifications_05679.csv"

# validation_data_path = "../data/simple_validation_data_05679"
# validation_data_classifications_path = "../data/simple_validation_classifications_05679.csv"

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

num_training_images = 10000
num_validation_images = 1000

num_channels = 3
image_height = 225
image_width = 400


############################################################################
################################### DATA ###################################
############################################################################

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



# def get_sampler(dataset, num_samples):
#     indices = np.random.choice(len(dataset), num_samples, replace=False)
#     return SubsetRandomSampler(indices)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        column_names = ['filename', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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

num_epochs = 100

learning_rate = 0.05

effective_batch_size = 50
actual_batch_size = 25
accumulation_steps = int(effective_batch_size / actual_batch_size)
aggregate_cost = 0

regularizer = "L2"
regularization_parameter = 0.01 / effective_batch_size

# To implement a variable learning rate schedule
batches_since_improvement = 0
learning_rate_decrease_threshold = (num_training_images / effective_batch_size) * 10
lowest_aggregate_cost_so_far = float('inf')


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
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    new_classifier = nn.Sequential(
        *list(model.classifier.children())[:6],  # Keep the first 6 layers of the original classifier. Note that the new classifier uses references to the layers of the original classifier. 
        nn.Linear(model.classifier[6].in_features, num_output_neurons),  # Final output layer
        nn.Sigmoid()
    )
    model.classifier = new_classifier.to(device)
else:
    model = NeuralNetwork().to(device)

# Use a basic stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization_parameter if regularizer == "L2" else 0)

if __name__ == '__main__':
    with open("../data/log.txt", 'w') as file:
        # Ensure we start with a clean slate for gradients
        optimizer.zero_grad(set_to_none=False)

        # Train for some number of epochs
        for epoch in range(num_epochs): 
            

            ############################################################################
            ################################# TRAINING #################################
            ############################################################################ 


            # sampler = get_sampler(training_data, num_training_images)
            # training_dataloader = DataLoader(training_data, batch_size=actual_batch_size, sampler=sampler)
            training_dataloader = DataLoader(training_data, batch_size=actual_batch_size, shuffle=True)
            mini_batch_number = 0
            for mini_batch, mini_batch_classifications in training_dataloader:
                print(f"epoch: {epoch} | mini_batch_number: {mini_batch_number} | learning_rate: {learning_rate} | batches_since_improvement: {batches_since_improvement}")
                
                # A 2-dimensional output. Each row is the predicted classification for a particular training image in the mini-batch
                training_prediction = model(mini_batch)

                # Define the cost function
                if cost_function == "quadratic":
                    cost = (((mini_batch_classifications - training_prediction) ** 2) / (2 * effective_batch_size)).sum()
                elif cost_function == "binary cross-entropy":
                    cost = nn.BCELoss()(training_prediction, mini_batch_classifications) / accumulation_steps

                aggregate_cost += cost

                # Backpropagation
                cost.backward()

                # Implement the effective batch size
                if (mini_batch_number + 1) % accumulation_steps == 0:
                    print(f"Aggregate cost: {aggregate_cost}")

                    file.write(f"epoch: {epoch} | mini_batch_number: {mini_batch_number} | learning_rate: {learning_rate} | batches_since_improvement: {batches_since_improvement} | aggregate_cost: {aggregate_cost}\n")

                    # Store the lowest cost so far in order to determine, after some number of batches of no improvement, when to decrease the learning rate while training
                    if aggregate_cost < lowest_aggregate_cost_so_far:
                        lowest_aggregate_cost_so_far = aggregate_cost 
                        batches_since_improvement = 0
                    else:
                        batches_since_improvement += 1

                    # Halve the learning rate when our aggregate cost hasn't decreased beyond what we've seen so far in some number of mini-batches
                    if batches_since_improvement >= learning_rate_decrease_threshold:
                        learning_rate /= 2
                        for g in optimizer.param_groups:
                            g["lr"] = learning_rate
                        
                        batches_since_improvement = 0
                        lowest_aggregate_cost_so_far = float('inf')

                    # Update weights and biases 
                    optimizer.step()

                    # Zero the gradients so they don't accumulate 
                    optimizer.zero_grad(set_to_none=False)

                    aggregate_cost = 0
                
                mini_batch_number += 1


            ############################################################################
            ################################ VALIDATION ################################
            ############################################################################ 


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

                correct += int(result)

            validation_accuracy = correct / num_validation_images

            print(f"Validation accuracy after epoch {epoch}: {validation_accuracy} ")

            file.write(f"Validation accuracy after epoch {epoch}: {validation_accuracy:.4f}\n")
            file.flush()

            torch.save(model, "model.pth")
