import os
import torch
from torch.utils.data import DataLoader

from neural_network import CustomImageDataset


def calculate_precision(confusion_matrix):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    num_classified_positives = TP + FP
    return TP / num_classified_positives if num_classified_positives != 0 else 0

def calculate_recall(confusion_matrix):
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    num_positives = TP + FN
    return TP / num_positives if num_positives != 0 else 0 


if __name__ == "__main__":
    validation_data_classifications_path = "../data/validation_classifications_1_error_small_grouping_5_and_7.csv"
    validation_data_path = "../data/validation_data_1_error_small_grouping_5_and_7"
    num_validation_images = len(os.listdir(validation_data_path))

    padding = len(str(num_validation_images))
    model_paths = ["model-log15/model8.pth"]
    num_labels = 9

    # The data
    validation_data = CustomImageDataset(
        validation_data_classifications_path,
        validation_data_path,
        num_labels
    )

    with open("../data/validation_accuracies.txt", 'w') as validation_accuracies_file, open("../data/incorrectly_classified_images.txt", 'w') as incorrectly_classified_images_file:
        incorrectly_classified_images_file.write(f"{''.rjust(padding)} | 0 1 2 3 4 5 6 7 8 9\n")
        for model_path in model_paths:
            model = torch.load(model_path)
            model.eval()

            validation_dataloader = DataLoader(validation_data, batch_size=1)

            results = [0 for _ in range(num_labels + 1)]

            # TP FP
            # FN TN
            confusion_matrix_list = [
                [[0, 0], 
                 [0, 0]] for _ in range(num_labels)
            ]

            images_processed = 0
            with torch.no_grad():
                for sample_validation_image, sample_validation_classification, image_index in validation_dataloader:
                    # A 1-D tensor with size (num_output_neurons,)
                    validation_prediction = model(sample_validation_image).squeeze()

                    # A 1-D tensor with size (num_output_neurons,)
                    sample_validation_classification = sample_validation_classification.squeeze()

                    # The absolute difference of the predicted labels from the actual labels 
                    difference = torch.abs(sample_validation_classification - validation_prediction)

                    # Create a boolean tensor where each element is True if the difference is less than 0.5
                    labels_correct = difference < 0.5

                    # Count the number of True values in the boolean tensor
                    num_labels_correct = torch.sum(labels_correct).item()

                    results[num_labels_correct] += 1

                    # The case where the model is not completely correct
                    if num_labels_correct != num_labels:
                        # Create a boolean tensor where each element is True if the difference is greater than 0.5
                        labels_incorrect = difference > 0.5
                        
                        model_incorrect_at = []
                        for label in range(len(labels_incorrect)):
                            if labels_incorrect[label]:
                                if sample_validation_classification[label] == 1:
                                    # The model outputs a 0 for this label when it should have output a 1; FN
                                    model_incorrect_at.append('0') 
                                    confusion_matrix_list[label][1][0] += 1
                                else:
                                    model_incorrect_at.append('1')  # The model outputs a 1 for this label when it should have output a 0; FP 
                                    confusion_matrix_list[label][0][1] += 1
                            else:
                                model_incorrect_at.append('.')

                        incorrectly_classified_images_file.write(f"{str(image_index.item()).rjust(padding)} | {" ".join(model_incorrect_at)}\n")
                    # The case where the model is completely correct
                    else:
                        for label, label_result in enumerate(sample_validation_classification):
                            # We have a true negative (TN) result
                            if label_result == 0:
                                confusion_matrix_list[label][1][1] += 1
                            # We have a true positive (TP) result
                            else:
                                confusion_matrix_list[label][0][0] += 1

                    images_processed += 1
                    if images_processed % 100 == 0:
                        print(f"{images_processed} images processed")
            
            for i, result in enumerate(results):
                validation_accuracies_file.write(f"{model_path} | Proportion of images for which exactly {i} of {num_labels} labels are correct: {result / num_validation_images:.4f}\n")
            
            for label, confusion_matrix in enumerate(confusion_matrix_list):
                precision = calculate_precision(confusion_matrix)
                recall = calculate_recall(confusion_matrix)
                validation_accuracies_file.write(f"{model_path} |        PRECISION of label {label}: {precision}\n")
                validation_accuracies_file.write(f"{model_path} |           RECALL of label {label}: {recall}\n")
                validation_accuracies_file.write(f"{model_path} | CONFUSION MATRIX of label {label}: {confusion_matrix}\n\n")

            validation_accuracies_file.flush()
