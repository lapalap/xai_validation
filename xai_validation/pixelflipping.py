import abc
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np  # type: ignore
import torch
import torch.nn as nn

from sklearn.metrics import roc_curve, auc # roc curve tools
from sklearn.metrics import precision_recall_curve

import cv2

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import math

def get_gaussian_kernel(kernel_size=5, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, padding = int((kernel_size-1)/2),
                                kernel_size=kernel_size, groups=channels, bias=False).cuda()

    #gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.data = (torch.ones_like(gaussian_kernel)/kernel_size**2).cuda()
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

gaussian_blur_standard = get_gaussian_kernel(kernel_size = 3, sigma = 4, channels = 1)
gaussian_blur_hard = get_gaussian_kernel(kernel_size = 3, sigma = 4, channels = 1)

def average_blur(img, size):
    image = np.array(img.transpose(0,1).transpose(1,2))
    image_blur = cv2.blur(image,(size,size))
    new_image = torch.tensor(image_blur).transpose(1,2).transpose(0,1)
    return new_image

def median_blur(img, size):
    image = np.array(img.transpose(0,1).transpose(1,2)).astype('float32') 
    image_blur = cv2.medianBlur(image,size)
    new_image = torch.tensor(image_blur).transpose(1,2).transpose(0,1)
    return new_image

class PerturbationPolicy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def perturb(self, input: torch.tensor, x: int, y: int):
      pass

    @abc.abstractmethod
    def initialise(self, input: torch.tensor, **args):
      pass


class RandomPermutation(PerturbationPolicy):
    def __init__(self, neighbourhood_size: int, input_size: Sequence[int], n_channels: int):
        self.neighbourhood_size = neighbourhood_size
        self.input_size = input_size
        self.n_channels = n_channels

    def initialise(self, input: torch.tensor):
      pass

    def perturb(self, input: torch.tensor, x: int, y: int):
        x_min = max(0, x - self.neighbourhood_size)
        x_max = min(self.input_size[0], x + self.neighbourhood_size)
        y_min = max(0, y - self.neighbourhood_size)
        y_max = min(self.input_size[1], y + self.neighbourhood_size)

        x_new = np.random.randint(x_min, x_max)
        y_new = np.random.randint(y_min, y_max)

        for c in range(self.n_channels):
            input[0][c][x][y] = input[0][c][x_new][y_new]


class RandomFill(PerturbationPolicy):
    def __init__(self, distribution: str, input_size: Sequence[int], n_channels: int):
        self.distribution = distribution
        self.input_size = input_size
        self.n_channels = n_channels
    
    def initialise(self, input: torch.tensor):
      pass

    def perturb(self, input: torch.tensor, x: int, y: int):

        if self.distribution == 'uniform':
            fill = np.random.uniform(size=self.n_channels)
        if self.distribution == 'normal':
            fill = np.random.normal(size=self.n_channels)
        if self.distribution == 'zeros':
            fill = np.zeros(self.n_channels)
        if self.distribution == 'ones':
            fill = np.ones(self.n_channels)

        for c in range(self.n_channels):
            input[0][c][x][y] = fill[c]

class GaussianBlur(PerturbationPolicy):
    def __init__(self, input_size: Sequence[int], n_channels: int,  kernel_size : int, sigma : int):
        self.input_size = input_size
        self.n_channels = n_channels

        self.kernel_size = kernel_size 
        self.sigma = sigma

        self.input = None
        self.blurred_input = None

    def initialise(self, input: torch.tensor):
        self.input = input

        kernel = get_gaussian_kernel(kernel_size = self.kernel_size, sigma = self.sigma, channels = self.n_channels)
        self.blurred_input = kernel(input).data


    def perturb(self, input: torch.tensor, x: int, y: int):
        for c in range(self.n_channels):
          input[0][c][x][y] = self.blurred_input[0][c][x][y]

class Composite(PerturbationPolicy):
    def __init__(self, neighbourhood_size: int, input_size: Sequence[int], n_channels: int,  kernel_size : int, sigma : int):

        self.neighbourhood_size = neighbourhood_size
        self.input_size = input_size
        self.n_channels = n_channels

        self.kernel_size = kernel_size 
        self.sigma = sigma

        self.input = None
        self.blurred_input = None

    def initialise(self, input: torch.tensor):
        self.input = input

        kernel = get_gaussian_kernel(kernel_size = self.kernel_size, sigma = self.sigma, channels = self.n_channels)
        self.blurred_input = kernel(input).data


    def perturb(self, input: torch.tensor, x: int, y: int):
        x_min = max(0, x - self.neighbourhood_size)
        x_max = min(self.input_size[0], x + self.neighbourhood_size)
        y_min = max(0, y - self.neighbourhood_size)
        y_max = min(self.input_size[1], y + self.neighbourhood_size)

        x_new = np.random.randint(x_min, x_max)
        y_new = np.random.randint(y_min, y_max)

        for c in range(self.n_channels):
            input[0][c][x][y] = self.blurred_input[0][c][x_new][y_new]


class Pixelflipping:
    """Class for performing a pixelflipping (input perturbation) of given explanation method.
    TODO: add documentation here
   """


    def __init__(self, model: nn.Module, input_dims: Sequence[int]) -> None:
      self.model = model
      self.input_dims = input_dims


    def run(self, input: torch.tensor, target_index: int, explanation: torch.tensor, method: PerturbationPolicy,
        step_size: int, n_steps: int):
        x = input.clone()

        sorted, index = explanation.view(-1).sort(descending=True)
        ind_x = index // self.input_dims[0]
        ind_y = index % self.input_dims[0]

        scores = torch.zeros([n_steps])

        method.initialise(input)
        with torch.no_grad():
            for i in range(n_steps):
                scores[i] = self.model(x)[0][target_index]
                # plt.imshow(x.cpu().view([3,32,32]).transpose(0,1).transpose(1,2))
                # plt.show()
                for j in range(step_size):
                    method.perturb(x, ind_x[i * step_size + j], ind_y[i * step_size + j])

        return scores
