import cv2
import numpy as np
import os

def load_images_and_annotations(data, input_size, grid):
    image_num = len(data['filename'].value_counts())
    num2classes = {i: class_name.strip() for i, class_name in enumerate(data["class"].unique())}
    classes2num = {class_name.strip(): i for i, class_name in enumerate(data["class"].unique())}
    num_classes = len(num2classes)
    annotations = np.zeros((image_num, grid[0], grid[1], 5+num_classes))
    images = np.zeros((image_num, input_size[0], input_size[1], 3))
    # Load the image
    filenames = data["filename"].unique()
    for i, filename in enumerate(filenames):
        image_path = os.path.join("/content/train", filename)
        image = cv2.imread(image_path)
        original_shape = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize the image to the input size expected by the YOLO model
        image = cv2.resize(image, input_size)
        # Normalize the image pixels to the range [0, 1]
        image = image / 255.0
        images[i] = image
        # Append the preprocessed image to the list
        # Split the line into components
        filename_annot = filename_annot = data[data['filename'] == filename]
        for index, row in filename_annot.iterrows():
            x_min, y_min, x_max, y_max = map(lambda x: float(x), row[['xmin', 'ymin', 'xmax', 'ymax']])
            x_min, x_max = map(lambda x: x / original_shape[1], [x_min, x_max])
            y_min, y_max = map(lambda x: x / original_shape[0], [y_min, y_max])
            # Extract the class label and bounding box coordinates
            class_label = classes2num[row['class']]
            # Append the normalized bounding box coordinates and class label
            # Calculate cell coordinates for bounding box
            cell_x_min, cell_y_min = round(((x_min + x_max)/2) * grid[0]), round(((y_min + y_max)/2) * grid[1])
            cell_x_min, cell_y_min = map(lambda x: x if x < grid[0] else grid[0]-1, [cell_x_min, cell_y_min])
            # Set bounding box info in the grid
            annotations[i, cell_y_min, cell_x_min, :4] = list(map(lambda x: x*input_size[0], [x_min, y_min, x_max, y_max]))
            annotations[i, cell_y_min, cell_x_min, 4] = 1  # Indicating the presence of an object
            annotations[i, cell_y_min, cell_x_min, 5+class_label] = 1  # Setting class label
    return images, annotations