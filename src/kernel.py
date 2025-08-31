import cv2
import numpy as np

# Using OpenCV (recommended)
def create_circular_kernel(outer_diameter, inner_diameter=0.0, gaussian_sigma=0.5, gaussian_kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_diameter, outer_diameter))
    kernel = kernel.astype(np.float64)

    inner_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_diameter, inner_diameter))
    inner_circle = inner_circle.astype(np.float64)

    padding_width = 20

    kernel = np.pad(kernel, pad_width=padding_width, mode='constant', constant_values=0)
    inner_circle = np.pad(inner_circle, pad_width=padding_width + (outer_diameter - inner_diameter) // 2, mode='constant', constant_values=0)
    
    kernel -= inner_circle

    return cv2.GaussianBlur(kernel, (gaussian_kernel_size, gaussian_kernel_size), gaussian_sigma)

def normalize_kernel(kernel):
    """Normalize kernel so sum of positive values = 1"""
    positive_sum = np.sum(kernel[kernel > 0])
    if positive_sum > 0:
        return kernel / positive_sum
    return kernel

# Create circular kernel using getStructuringElement
''' Really good one'''
larger_circle = -create_circular_kernel(
    outer_diameter=41, 
    inner_diameter=31,     # 41 * 0.37
    gaussian_sigma=1.1,
    gaussian_kernel_size=13
) / 2

smaller_circle = create_circular_kernel(
    outer_diameter=21, 
    inner_diameter=15,     # 41 * 0.37
    gaussian_sigma=1.2,
    gaussian_kernel_size=15
)


''' config 2 - also interesting
larger_circle = -create_circular_kernel(
    outer_diameter=141, 
    inner_diameter=91,     # 41 * 0.37
    gaussian_sigma=10,
    gaussian_kernel_size=13
) / 5

smaller_circle = create_circular_kernel(
    outer_diameter=41, 
    inner_diameter=15,     # 41 * 0.37
    gaussian_sigma=0,
    gaussian_kernel_size=15
)


larger_circle = -create_circular_kernel(
    outer_diameter=141, 
    inner_diameter=91,     # 41 * 0.37
    gaussian_sigma=10,
    gaussian_kernel_size=13
) / 2

smaller_circle = create_circular_kernel(
    outer_diameter=41, 
    inner_diameter=15,     # 41 * 0.37
    gaussian_sigma=1,
    gaussian_kernel_size=15
)'''

padding = (larger_circle.shape[0] - smaller_circle.shape[0]) // 2
smaller_circle = np.pad(smaller_circle, pad_width=padding, mode='constant', constant_values=0)

kernel = normalize_kernel(larger_circle + smaller_circle)

