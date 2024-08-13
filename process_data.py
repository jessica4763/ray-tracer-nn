import csv
from imageio import imread
import numpy as np
import os
from random import choice
import torch
import matplotlib.pyplot as plt

training_data_path = "../data/simple_training_data_05679"
training_data_classifications_path = "../data/simple_training_classifications_05679.csv"

validation_data_path = "../data/simple_validation_data_05679"
validation_data_classifications_path = "../data/simple_validation_classifications_05679.csv"

training_image_names = os.listdir(training_data_path)
validation_image_names = os.listdir(validation_data_path)


def get_image_classifications(image_classification_path):
    # Associate the data images with their classifications
    image_classifications = {}
    with open(image_classification_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_name = row[0]
            image_classification = [int(c) for c in row[1:]]
            image_classifications[image_name] = image_classification

    return image_classifications


# Associate the training and validation data images with their classifications
training_classifications = get_image_classifications(training_data_classifications_path)
validation_classifications = get_image_classifications(validation_data_classifications_path)

# Take a random sample of training data images and display them with their name and classification
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_training_image_name = choice(training_image_names)
#     sample_training_image_path = os.path.join(training_data_path, sample_training_image_name)
#     sample_training_image_classification = training_classifications[sample_training_image_name]
#     sample_training_image = imread(sample_training_image_path, format="ppm")
#     figure.add_subplot(rows, cols, i)
#     plt.title(f"{sample_training_image_name}: {sample_training_image_classification}", fontsize=8)
#     plt.axis("off")
#     plt.imshow(sample_training_image)
# plt.show()
