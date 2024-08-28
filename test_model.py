import os
import torch
from torch.utils.data import DataLoader

from neural_network import CustomImageDataset


if __name__ == "__main__":
    validation_data_classifications_path = "../data/validation_classifications.csv"
    validation_data_path = "../data/validation_data"
    num_validation_images = len(os.listdir(validation_data_path))
    validation_data = CustomImageDataset(
        validation_data_classifications_path,
        validation_data_path,
    )

    model_names = ["model-log3", "model-log4", "model-log5", "model-log6"]

    with open("../data/validation_accuracies.txt", 'w') as file:
        for model_name in model_names:
            model_path = f"{model_name}.pth"
            model = torch.load(model_path)
            model.eval()

            # sampler = get_sampler(validation_data, num_validation_images)
            # validation_dataloader = DataLoader(validation_data, batch_size=1, sampler=sampler)
            validation_dataloader = DataLoader(validation_data, batch_size=1)

            results = [0 for _ in range(11)]

            i = 0

            with torch.no_grad():
                for sample_validation_image, sample_validation_classification in validation_dataloader:
                    # A 1-D tensor with size (num_output_neurons,)
                    validation_prediction = model(sample_validation_image).squeeze()
                    
                    # A 1-D tensor with size (num_output_neurons,)
                    sample_validation_classification = sample_validation_classification.squeeze()

                    # The absolute difference of the predicted labels from the actual labels actual labels 
                    difference = torch.abs(sample_validation_classification - validation_prediction)

                    # Create a boolean tensor where each element is True if the difference is less than 0.5
                    labels_correct = difference < 0.5

                    # Count the number of True values in the boolean tensor
                    num_labels_correct = torch.sum(labels_correct).item()

                    results[num_labels_correct] += 1

                    i += 1
                    if i % 100 == 0:
                        print(f"{i} images done")
            
            for i, result in enumerate(results):
                file.write(f"{model_name} for {i} of 10 images: {result / num_validation_images:.4f}\n")

            file.flush()
