# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
    )
)

from helpers.dataloader import load_image, get_data_path


SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"


def read_image():
    """
    Load and preprocess the Sphinx image:
    scale by 0.5 and convert to greyscale.
    """
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(
        image_path,
        scale_factor=2,
        as_gray=True
    )


class GaussianFilt:

    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """
        Generate a normalised 1D Gaussian kernel.

        Returns:
            gauss_1d: ndarray of shape (k_size, 1)
        """

        # Kernel size required by the lab
        k_size = int(6 * self.sigma + 1)

        # Centre position
        centre = k_size // 2

        # Integer positions around the centre
        x = np.arange(
            -centre,
            centre + 1,
            dtype=np.float64
        )

        # Gaussian function
        kernel = np.exp(
            -(x ** 2) /
            (2 * self.sigma ** 2)
        )

        # Normalise the kernel so that its sum is 1
        kernel = kernel / np.sum(kernel)

        # Return as a column vector
        kernel = np.expand_dims(
            kernel,
            axis=1
        )

        return kernel

    def my_conv_method(self, image):
        """
        Perform efficient separable 2D Gaussian convolution.

        The Gaussian filter is separable, so the 2D convolution
        is performed using two 1D convolution passes:
        horizontal followed by vertical.

        Args:
            image: 2D greyscale image.

        Returns:
            Blurred image with the same shape as input
            and dtype uint8.
        """

        # Get the 1D Gaussian kernel
        kernel = self.gauss_kernel()[:, 0]

        # Kernel size
        k_size = len(kernel)

        # Amount of padding required
        pad = k_size // 2

        # Convert image to float for convolution
        image = image.astype(np.float64)

        # Pad image once

        padded = np.pad(
            image,
            (
                (pad, pad),
                (pad, pad)
            ),
            mode="constant",
            constant_values=0
        )

        # First pass: horizontal convolution

        windows_h = np.lib.stride_tricks.sliding_window_view(
            padded,
            k_size,
            axis=1
        )

        horizontal = np.sum(
            windows_h *
            kernel.reshape(1, 1, -1),
            axis=-1
        )

        # Second pass: vertical convolution

        windows_v = np.lib.stride_tricks.sliding_window_view(
            horizontal,
            k_size,
            axis=0
        )

        vertical = np.sum(
            windows_v *
            kernel.reshape(1, 1, -1),
            axis=-1
        )

        # Convert result to valid image range

        vertical = np.clip(
            vertical,
            0,
            255
        )

        # Required output type
        vertical = vertical.astype(
            np.uint8
        )

        return vertical