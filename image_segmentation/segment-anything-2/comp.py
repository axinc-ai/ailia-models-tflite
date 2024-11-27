

import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_accuracy(mask1, mask2):
  """
  Calculates the accuracy between two segmentation masks.

  Args:
    mask1: The first segmentation mask as a NumPy array.
    mask2: The second segmentation mask as a NumPy array.

  Returns:
    The accuracy as a float between 0 and 1.
  """

  # Ensure masks are of the same shape and data type
  if mask1.shape != mask2.shape:
    raise ValueError("Masks must have the same shape.")
  if mask1.dtype != mask2.dtype:
    raise ValueError("Masks must have the same data type.")

  # Calculate the number of correctly classified pixels
  num_correct = np.sum(mask1 == mask2)

  # Calculate the total number of pixels
  total_pixels = mask1.size

  # Calculate the accuracy
  accuracy = num_correct / total_pixels

  return accuracy


def calculate_iou(mask1, mask2):
  """
  Calculates the Intersection over Union (IoU) between two segmentation masks.

  Args:
    mask1: The first segmentation mask as a NumPy array.
    mask2: The second segmentation mask as a NumPy array.

  Returns:
    The IoU as a float between 0 and 1.
  """

  # Ensure masks are of the same shape and data type
  if mask1.shape != mask2.shape:
    raise ValueError("Masks must have the same shape.")
  if mask1.dtype != mask2.dtype:
    raise ValueError("Masks must have the same data type.")

  # Calculate the intersection
  intersection = np.logical_and(mask1, mask2)

  # Calculate the union
  union = np.logical_or(mask1, mask2)

  # Calculate IoU
  iou = np.sum(intersection) / np.sum(union)

  return iou 



def calculate_dice(mask1, mask2):
  """
  Calculates the Dice coefficient between two segmentation masks.

  Args:
    mask1: The first segmentation mask as a NumPy array.
    mask2: The second segmentation mask as a NumPy array.

  Returns:
    The Dice coefficient as a float between 0 and 1.
  """

  # Ensure masks are of the same shape and data type
  if mask1.shape != mask2.shape:
    raise ValueError("Masks must have the same shape.")
  if mask1.dtype != mask2.dtype:
    raise ValueError("Masks must have the same data type.")

  # Calculate the intersection
  intersection = np.logical_and(mask1, mask2)

  # Calculate the Dice coefficient
  dice = 2 * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2))

  return dice

def visualize_masks(image, mask1, mask2, save_path=None):
  """
  Visualizes the original image, ground truth mask, and predicted mask.

  Args:
    image: The original image as a NumPy array.
    mask1: The ground truth mask as a NumPy array.
    mask2: The predicted mask as a NumPy array.
    save_path: (Optional) The path to save the figure. If None, the figure is not saved.
  """

  plt.figure(figsize=(10, 5))

  plt.subplot(1, 3, 1)
  plt.imshow(image)
  plt.title("Original Image")

  plt.subplot(1, 3, 2)
  plt.imshow(mask1, cmap='gray')
  plt.title("Ground Truth Mask")

  plt.subplot(1, 3, 3)
  plt.imshow(mask2, cmap='gray')
  plt.title("Predicted Mask")

  if save_path is not None:
    plt.savefig(save_path)

  plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_difference_map(image, mask1, mask2, symmetric=False, save_path=None):
    """
    Visualizes the difference map between two masks and the original image.
    
    Args:
        image: Original image.
        mask1: Ground truth mask.
        mask2: Predicted mask.
        symmetric: If True, use symmetric quantization title, otherwise asymmetric.
        save_path: (Optional) Path to save the figure.
    """

    # Calculate the difference map using bitwise_xor (^)
    difference_map = mask1 ^ mask2
    color = np.array([0, 255, 0])
    color = color.reshape(1, 1, -1)
    mask_image = difference_map * color
    img = (image * ~difference_map) + (image * difference_map) * 0.6 + mask_image * 0.4

    # Optionally save the modified image
    if save_path:
        cv2.imwrite("sft" + save_path, img)

    # Set up a 2x2 grid for subplots
    plt.figure(figsize=(10, 10))

    # First row
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(difference_map, cmap='jet')
    plt.title("Difference Map")
    colorbar = plt.colorbar(shrink=0.7)  # Make colorbar smaller

    # Second row
    plt.subplot(2, 2, 3)
    plt.imshow(mask1, cmap='gray')
    plt.title("No quantization")

    plt.subplot(2, 2, 4)
    plt.imshow(mask2, cmap='gray')
    plt.title("Symmetric quantization" if symmetric else "Asymmetric quantization")

    # Adjust layout
    plt.tight_layout()

    # Save and display the figure
    if save_path:
        plt.savefig(save_path)
    plt.show()


from PIL import Image
def main():
    imgs=["truck.jpg", "dog.jpg", "dog.jpg", "groceries.jpg"]


    original_outputs_mask = [
    "output_car_original.jpg.npy",
    "output_dog_original2.jpg.npy",
    "output_dog_original.jpg.npy",
    "output_groceries_original.jpg.npy"
    ]
    symmetric_outputs_mask = [
        "output_car_quantized.jpg.npy",
        "output_dog_quantized2.jpg.npy",
        "output_dog_quantized.jpg.npy",
        "output_groceries_quantized.jpg.npy"
    ]

    assymetric_outputs_mask = [
        "output_car_quantized_assymmetric.jpg.npy",
        "output_dog_quantized2_assymmetric.jpg.npy",
        "output_dog_assymetric_quantized_ear.jpg.npy",
        "output_groceries_quantized_assymmetric.jpg.npy"
    ]

    print("Comparision of float and symmetric outputs")
    for i in range(4):
        image = Image.open(imgs[i])
        # Convert the image to a NumPy array
        image = np.array(image)
        mask1 = np.load(original_outputs_mask[i])
        mask2 = np.load(symmetric_outputs_mask[i])
        print("Current image name: ", imgs[i])
        print("Accuracy: ", calculate_accuracy(mask1, mask2))
        print("IOU: ", calculate_iou(mask1, mask2))
        print("Dice: ", calculate_dice(mask1, mask2))

        visualize_difference_map(image, mask1, mask2, symmetric=True, save_path="symmetric_comp"+str(i)+".png")
        print("")
        print("")

    print("___________________________________________________________________________________________")
    print("Comparision of float and asymmetric outputs")
    for i in range(4):
        image = Image.open(imgs[i])
        # Convert the image to a NumPy array
        image = np.array(image)
        mask1 = np.load(original_outputs_mask[i])
        mask2 = np.load(assymetric_outputs_mask[i])
        print("Current image name: ", imgs[i])
        print("Accuracy: ", calculate_accuracy(mask1, mask2))
        print("IOU: ", calculate_iou(mask1, mask2))
        print("Dice: ", calculate_dice(mask1, mask2))

        visualize_difference_map(image, mask1, mask2, save_path="asymmetric_comp"+str(i)+".png")
        print("")
        print("")



if __name__ == '__main__':
    main()