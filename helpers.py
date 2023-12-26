import matplotlib.pyplot as plt
import numpy as np

def plot_image_with_boxes(image, annotations, grid, num2classes):
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for h in range(grid[0]):
        for w in range(grid[1]):
            xmin, ymin, xmax, ymax = annotations[h, w, :4]
            width, height = xmax - xmin, ymax - ymin
            confidence = annotations[h, w, 4]
            if confidence < 0.4:
                continue
            # Find the predicted class label
            predicted_class = num2classes[np.argmax(annotations[h, w, 5:])]
            # Draw the bounding box rectangle
            rect = plt.Rectangle((xmin, ymin), width, height, fill=False, edgecolor='r', linewidth=2)
            ax.add_patch(rect)

            # Add the class label text
            ax.text(xmin, ymin, f'{predicted_class}', color='r', fontsize=10, backgroundcolor='k')

    # Show the plot
    plt.show()