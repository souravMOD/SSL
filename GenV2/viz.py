import cv2
import matplotlib.pyplot as plt
import os
from ipywidgets import interact, IntSlider
import ipywidgets as widgets

# Function to display YOLO annotations on an image
def visualize_yolo_annotations(image_path, annotations_path, class_names=None):
    # Load the image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Open the annotations file
    with open(annotations_path, 'r') as f:
        lines = f.readlines()

    # Loop over each line in the annotation file
    for line in lines:
        obj_class, x_center, y_center, width, height = map(float, line.split())

        # Convert YOLO format (normalized) to pixel values
        x_center_pixel = int(x_center * img_width)
        y_center_pixel = int(y_center * img_height)
        width_pixel = int(width * img_width)
        height_pixel = int(height * img_height)

        # Calculate the top-left corner of the bounding box
        x_min = int(x_center_pixel - width_pixel / 2)
        y_min = int(y_center_pixel - height_pixel / 2)
        x_max = int(x_center_pixel + width_pixel / 2)
        y_max = int(y_center_pixel + height_pixel / 2)

        # Draw the bounding box on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Add label if class names are provided
        if class_names:
            label = f"{class_names[int(obj_class)]}"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Function to process a folder of images and annotations with slider
def process_folder_with_slider(images_folder, annotations_folder, class_names=None):
    # Get the list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Sort the image files for better navigation
    image_files.sort()

    def show_image(index):
        image_file = image_files[index]
        image_path = os.path.join(images_folder, image_file)

        # Get the corresponding annotation file
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotations_path = os.path.join(annotations_folder, annotation_file)

        # Check if annotation file exists
        if os.path.exists(annotations_path):
            visualize_yolo_annotations(image_path, annotations_path, class_names)
        else:
            print(f"Annotation file not found for {image_file}")

    # Create a slider to navigate through images
    interact(show_image, index=IntSlider(min=0, max=len(image_files)-1, step=1, description="Image"))

# Usage example:
images_folder = r"D:\GenV2\train\images"
annotations_folder = r"D:\GenV2\train\labels"
class_names = ['Dicot', 'Monocot', 'Sugarbeet']  # Modify as per your dataset

# Process the folder with a slider
process_folder_with_slider(images_folder, annotations_folder, class_names)
