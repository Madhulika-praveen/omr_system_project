# src/omr/preprocessing.py

import cv2
import numpy as np

def preprocess_sheet(img: np.ndarray) -> np.ndarray:
    """
    Dummy preprocessing function.
    Replace with actual OMR preprocessing logic later.
    """
    # Example: convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray
