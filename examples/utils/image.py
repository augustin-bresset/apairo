"""Image visualization and manipulation functions."""

import numpy as np
from matplotlib import pyplot as plt

def plt_image(image):
    """Plot an image.

    Args:
        image (np.ndarray) : The image to plot
    """
    plt.imshow(image, cmap='gray')
    plt.show()

def plt_image_scaled(image, v_scale=5, h_scale=5, show=True):
    """Plot an image with a scale on the image.

    Args:
        image (np.ndarray) : The image to plot
        v_scale (int) : The number of vertical scale
        h_scale (int) : The number of horizontal scale
    """
    plt.imshow(image, cmap='gray')
    horizontal_scale = np.linspace(0, image.shape[0], v_scale)
    vertical_scale = np.linspace(0, image.shape[1], h_scale)
    plt.yticks(horizontal_scale)
    plt.xticks(vertical_scale)
    plt.grid()
    if show:
        plt.show()

def sub_rect_image(image, x, y, width, height, out=None):
    """Return a sub image of the image with the size given.

    The sub image is defined by the top left corner (x, y) and the size (width, height).
    
    Args:
        image (np.ndarray) : The image to crop
        x (int) : The x coordinate of the top left corner
        y (int) : The y coordinate of the top left corner
        width (int) : The width of the sub image
        height (int) : The height of the sub image

    Returns:
        np.ndarray : The sub image
    """
    if out is not None:
        return image[y:y + height, x:x + width]

    out = image[y:y + height, x:x + width] 
    return out

def sub_centered_low_image(image, width, height, out=None):
    """Return a sub image of the image with the size given.

    The sub image is centered for width and takes the bottom of the image for height.
    
    Args:
        image (np.ndarray) : The image to crop
        width (int) : The width of the sub image
        height (int) : The height of the sub image

    Returns:
        np.ndarray : The sub image
    """
    if isinstance(width, float):
        width = int(image.shape[1]*width)
    if isinstance(height, float):
        height = int(image.shape[0]*height)

    x = (image.shape[1] - width) // 2
    y = image.shape[0] - height
    return sub_rect_image(image, x, y, width, height, out=out)