from imageio import imread
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import models


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
print(f"Using {device} device")


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

        return image, labels, idx


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


if __name__ == '__main__':


    ############################################################################
    ################################### DATA ###################################
    ############################################################################


    training_data_classifications_path = "../data/training_classifications_1_error_small.csv"
    training_data_path = "../data/training_data_1_error_small"

    validation_data_classifications_path = "../data/validation_classifications_1_error_small.csv"
    validation_data_path = "../data/validation_data_1_error_small"

    num_training_images = len(os.listdir(training_data_path))
    num_validation_images = len(os.listdir(validation_data_path))

    num_channels = 3
    image_height = 225
    image_width = 400

    labels_map = {
        0: "Did not normalise point to light vector.",
        1: "Did not normalise view vector.",
        2: "Did not apply the max function in the diffuse calculation.",
        3: "Did not apply the max function in the specular calculation.",
        4: "Applied the pow function inside the max function.",
        5: "Did not negate the point to light normalised vector inside reflect.",
        6: "Calculated a light to point vector instead of a point to light vector.",
        7: "Calculated a camera to point vector instead of a point to camera vector.",
        8: "Did not consider light colour* and that light intensity decreases with distance in specular calcluation.",
        9: "Did not nudge shadow ray.",
    }

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

    num_output_neurons = len(labels_map)

    # Define the neural network to be used
    network = "VGG-16"

    # Define the loss function to be used
    loss_function = "binary cross-entropy loss"

    num_epochs = 100

    learning_rate = 0.05

    # To implement momentum-based gradient descent 
    momentum_coefficient = 0

    # To implement a larger effective batch size
    effective_batch_size = 10
    actual_batch_size = 10
    accumulation_steps = int(effective_batch_size / actual_batch_size)
    aggregate_training_loss = 0

    # To implement a variable learning rate schedule
    batches_since_improvement = 0
    batches_per_epoch = num_training_images / effective_batch_size
    learning_rate_decrease_threshold = batches_per_epoch * 10
    lowest_aggregate_training_loss_so_far = float('inf')

    regularizer = "L2"
    regularization_parameter = 0.01 / effective_batch_size

    if network == "VGG-16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        new_classifier = nn.Sequential(
            *list(model.classifier.children())[:6],  # Keep the first 6 layers of the origifnal classifier. Note that the new classifier uses references to the layers of the original classifier. 
            nn.Linear(model.classifier[6].in_features, num_output_neurons),  # Final output layer
            nn.LogSoftmax() if loss_function == "softmax with negative log-likelihood loss" else nn.Sigmoid()
        )
        model.classifier = new_classifier.to(device)
    else:
        model = NeuralNetwork().to(device)

    # Use a basic stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization_parameter if regularizer == "L2" else 0, momentum=momentum_coefficient)

    # Ensure we start with a clean slate for gradients
    optimizer.zero_grad(set_to_none=False)

    with open("../data/log.txt", 'w') as file:
        # Train for some number of epochs
        for epoch in range(num_epochs): 
            

            ############################################################################
            ################################# TRAINING #################################
            ############################################################################ 


            model.train()

            # sampler = get_sampler(training_data, num_training_images)
            # training_dataloader = DataLoader(training_data, batch_size=actual_batch_size, sampler=sampler)
            training_dataloader = DataLoader(training_data, batch_size=actual_batch_size, shuffle=True)
            mini_batch_number = 0

            for mini_batch, mini_batch_classifications, _ in training_dataloader:
                print(f"epoch: {epoch} | mini_batch_number: {mini_batch_number} | learning_rate: {learning_rate} | batches_since_improvement: {batches_since_improvement}")
                
                # A 2-dimensional output. Each row is the predicted classification for a particular training image in the mini-batch
                training_prediction = model(mini_batch)

                # Training loss
                if loss_function == "quadratic loss":
                    training_loss = (((mini_batch_classifications - training_prediction) ** 2) / (2 * effective_batch_size)).sum()
                elif loss_function == "binary cross-entropy loss":
                    training_loss = nn.BCELoss()(training_prediction, mini_batch_classifications) / accumulation_steps
                elif loss_function == "softmax with negative log-likelihood loss":
                    mini_batch_classifications = mini_batch_classifications.argmax(dim=1)
                    training_loss = nn.NLLLoss()(training_prediction, mini_batch_classifications)

                aggregate_training_loss += training_loss

                # Backpropagation
                training_loss.backward()

                # Implement the effective batch size
                if (mini_batch_number + 1) % accumulation_steps == 0:
                    print(f"Aggregate training loss: {aggregate_training_loss}")

                    file.write(f"epoch: {epoch} | mini_batch_number: {mini_batch_number} | learning_rate: {learning_rate} | batches_since_improvement: {batches_since_improvement} | aggregate_training_loss: {aggregate_training_loss}\n")

                    # Store the lowest training loss so far in order to determine, after some number of batches of no improvement, when to decrease the learning rate while training
                    if aggregate_training_loss < lowest_aggregate_training_loss_so_far:
                        lowest_aggregate_training_loss_so_far = aggregate_training_loss 
                        batches_since_improvement = 0
                    else:
                        batches_since_improvement += 1

                    # Halve the learning rate when our aggregate loss hasn't decreased beyond what we've seen so far in some number of mini-batches
                    if batches_since_improvement >= learning_rate_decrease_threshold:
                        learning_rate /= 2
                        for g in optimizer.param_groups:
                            g["lr"] = learning_rate
                        
                        batches_since_improvement = 0
                        lowest_aggregate_training_loss_so_far = float('inf')

                    # Update weights and biases 
                    optimizer.step()

                    # Zero the gradients so they don't accumulate 
                    optimizer.zero_grad(set_to_none=False)

                    aggregate_training_loss = 0
                
                mini_batch_number += 1


            ############################################################################
            ################################ VALIDATION ################################
            ############################################################################ 


            model.eval()

            # sampler = get_sampler(validation_data, num_validation_images)
            # validation_dataloader = DataLoader(validation_data, batch_size=1, sampler=sampler)
            validation_dataloader = DataLoader(validation_data, batch_size=1)
            aggregate_validation_loss = 0
            results = [0 for _ in range(11)]

            with torch.no_grad():
                for sample_validation_image, sample_validation_classification, _ in validation_dataloader:
                    # A 1-D tensor with size (num_output_neurons,)
                    validation_prediction = model(sample_validation_image).squeeze()
                    sample_validation_classification = sample_validation_classification.squeeze()

                    # Validation loss for reporting (using the same loss function as for training loss)
                    if loss_function == "quadratic loss":
                        validation_loss = (((sample_validation_classification - validation_prediction) ** 2) / 2).sum()
                    elif loss_function == "binary cross-entropy loss":
                        validation_loss = nn.BCELoss()(validation_prediction, sample_validation_classification)
                    elif loss_function == "softmax with negative log-likelihood loss":
                        sample_validation_classification = sample_validation_classification.unsqueeze(0).argmax()
                        validation_loss = nn.NLLLoss()(validation_prediction, sample_validation_classification)

                    aggregate_validation_loss += validation_loss

                    # A 1-D tensor with size (num_output_neurons,)
                    sample_validation_classification = sample_validation_classification.squeeze()

                    # The prediction is correct if all predicted labels are within 0.5 of the actual labels 
                    difference = torch.abs(sample_validation_classification - validation_prediction)

                    # Create a boolean tensor where each element is True if the difference is less than 0.5
                    labels_correct = difference < 0.5

                    # Count the number of True values in the boolean tensor
                    num_labels_correct = torch.sum(labels_correct).item()

                    results[num_labels_correct] += 1

            for i, result in enumerate(results):
                file.write(f" Epoch {epoch} | Proportion of images for which exactly {i} of 10 labels are correct: {result / num_validation_images:.4f}\n")

            average_validation_loss = aggregate_validation_loss / num_validation_images
            file.write(f"Validation loss after epoch {epoch}: {average_validation_loss}\n")

            file.flush()

            torch.save(model, f"model{epoch}.pth")
