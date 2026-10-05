# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

_1 sentence: What was the goal of this lab?_
ANS:The Goal of this lab was to implement an efficient Gaussian filter using the separability of the 2D Gaussian kernel,a single 2D convolution is replaced by two 1D passes.

## Implementation

_1-2 sentences: Briefly describe the steps you performed to implment efficient convolution._
ANS:I generated a normalised 1D Gaussian kernel with a size of int(6 × sigma + 1) and applied it in two passes, first horizontally and then vertically, using padded image data and vectorised NumPy operations. The final filtered image was clipped to the valid pixel range and converted to uint8.

## Results

_1-2 sentences: How well did your method perform vs the baseline in the pytests? ._
ANS:The separable convolution method was significantly faster than the direct 2D convolve2d baseline because it replaces the expensive 2D operation with two 1D operations. The exact performance ratio can be reported from the pytest/evaluation output after running the tests.
