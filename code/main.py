from code.dataset import CoinDataset
from code.evaluation.evaluation import evaluate_dataset
from code.model.model import model_test

def main():
  # Define paths to your image and annotation directories
  image_dir = "dataset/images"
  annotation_dir = "dataset/labels"

  # Create a CoinDataset instance
  dataset = CoinDataset(image_dir, annotation_dir)

# Access the validation set using indexing (assuming validation set is stored in `val_images` and `val_annotations`)
  validation_images = dataset.val_images
  validation_annotations = dataset.val_annotations

# Evaluate the model on the validation set
  evaluation_results = evaluate_dataset(validation_images, validation_annotations)

# Print the evaluation results
  print("Validation Set Evaluation:")
  for metric, value in evaluation_results.items():
    print(f"\t{metric}: {value}")

"""
  # Print evaluation results
  print("Evaluation Results:")
  for key, value in evaluation_results.items():
    print(f"{key}: {value}")

"""

if __name__ == "__main__":
    main()



