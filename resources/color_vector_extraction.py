import numpy as np
import cv2

def color_histogram(image, bins=256):
    """
    Compute the color histogram of an RGB image.

    Parameters:
    - image (numpy.ndarray): Input RGB image represented as a numpy array.
    - bins (int): Number of bins for the histogram (default is 256).

    Returns:
    numpy.ndarray: Concatenated color histogram of the input image.
    """

    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    
    # Normalize
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()
    
    # Merging the histograms
    histogram = np.concatenate([hist_r, hist_g, hist_b])
    
    return histogram