import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import stats
import matplotlib.pyplot as plt


def clamp_matrix(processed_matrix):
    """
    Rounds a value to -1 or 1 if it's not within the range [-1, 1].
    """
    # Find elements outside the range [-1, 1]
    mask_lower = processed_matrix < -1
    mask_upper = processed_matrix > 1

    # Clamp elements to the range [-1, 1]
    processed_matrix[mask_lower] = -1
    processed_matrix[mask_upper] = 1

    # Round elements to the nearest integer
    # processed_matrix = np.round(processed_matrix)

    return processed_matrix


def normalize(numbers):
    print numbers
    # Find the minimum and maximum values in the list
    min_val = min(numbers)
    max_val = max(numbers)

    # Normalize each number in the list
    normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]

    print("Original numbers:", numbers)
    print("Normalized numbers:", normalized_numbers)
    return normalized_numbers


def scale_to_range(float_list, target_min=0, target_max=10):
    """
    Maps a list of floats to the range [target_min, target_max].

    Parameters:
    float_list (list): A list of float values.
    target_min (float): The minimum value of the target range.
    target_max (float): The maximum value of the target range.

    Returns:
    list: A new list with the float values mapped to the target range.
    """
    # Find the minimum and maximum values in the input list
    input_min = min(float_list)
    input_max = max(float_list)

    # Map the input values to the target range
    scaled_list = [(x - input_min) / (input_max - input_min) * (target_max - target_min) + target_min for x in
                   float_list]

    return scaled_list


def generate_n_copies_of_time_list(time_list, n):
    # Create an empty list to store the result
    result_list = []

    # Extend the result_list with n copies of the original_list
    for _ in range(n):
        result_list.append(time_list)

    return result_list


def get_timestamps(fps, frame_number):
    """
    Generates a list of timestamps based on the given frames per second and frame number.

    Args:
    fps (int): The frames per second.
    frame_number (int): The frame number.

    Returns:
    list: A list of timestamps, starting from 1 second.
    """
    timestamps = []
    for i in range(1, frame_number + 1):
        timestamp = float(i) / fps
        timestamps.append(timestamp)
    return timestamps


def remove_redundant_frames(obj):
    directional_vecs = obj['out_dir_vec']
    human_vecs = obj['human_dir_vec']
    print "length of directional vecs =", directional_vecs.shape[0]
    print "length of human vecs =", human_vecs.shape[0]

    directional_vecs = directional_vecs[:human_vecs.shape[0]]
    print "length of directional vecs after removing redundant =", directional_vecs.shape[0]

    return directional_vecs


def savgol_filter_smooth_radian_list(names, radians_list, window_size, polyorder, visualize):
    # Apply a filter to smooth the angles
    smoothed_radians_list = []

    for i in range(len(radians_list)):
        name_temp = names[i]
        radian_temp = radians_list[i]
        smoothed_radians = savgol_filter(radian_temp, window_size, polyorder)
        smoothed_radians_float_list = [float(i) for i in smoothed_radians.tolist()]
        smoothed_radians_list.append(smoothed_radians_float_list)
        if visualize:
            visualize_smooth(name_temp, radian_temp, smoothed_radians)

    return smoothed_radians_list


def handle_outlier(names, radians_list, threshold, visualize):
    # Apply a handle outlier function to the radian_list
    smoothed_radians_list = []

    for i in range(len(radians_list)):
        radian_temp = copy.deepcopy(radians_list[i])
        name_temp = names[i]
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(radian_temp))
        outlier_indices = np.where(z_scores > threshold)[0]

        # Replace outliers with median
        median_angle = float(np.median(radian_temp))
        # print i, outlier_indices, "median_angle =", median_angle

        # print "median_angle_type =", type(median_angle)
        # print "array_element_type =", type(radian_temp[0])

        for index in outlier_indices:
            radian_temp[index] = median_angle

        smoothed_radians_list.append(radian_temp)

        if visualize:
            visualize_smooth(name_temp, radians_list[i], radian_temp)

    return smoothed_radians_list


def moving_average(names, radians_list, window_size, visualize):
    # Apply a handle outlier function to the radian_list
    smoothed_radians_list = []

    for i in range(len(radians_list)):
        radian_temp = radians_list[i]
        name_temp = names[i]
        window = np.ones(int(window_size)) / float(window_size)
        smoothed_radians = np.convolve(radian_temp, window, 'same')

        smoothed_radians_float_list = [float(i) for i in smoothed_radians.tolist()]

        smoothed_radians_list.append(smoothed_radians_float_list)

        if visualize:
            visualize_smooth(name_temp, radian_temp, smoothed_radians)

    return smoothed_radians_list


def visualize_smooth(name_temp, angles, smoothed_angles):
    plt.figure(figsize=(10, 5))
    plt.plot(angles, label='Original Data')
    plt.plot(smoothed_angles, label='Smoothed Data', color='red')
    plt.legend()
    plt.title(name_temp + 'Angle Data Smoothing')
    plt.show()


def exponential_moving_average_smooth(names, radians_list, alpha, visualize):
    # Apply a handle outlier function to the radian_list
    smoothed_radians_list = []
    for i in range(len(radians_list)):
        radian_temp = radians_list[i]
        name_temp = names[i]
        smoothed_radians = exponential_moving_average(name_temp, radian_temp, alpha, visualize)

        smoothed_radians_float_list = [float(i) for i in smoothed_radians.tolist()]

        smoothed_radians_list.append(smoothed_radians_float_list)

    return smoothed_radians_list


def exponential_moving_average(name, data, alpha, visualize):
    """
    Compute the Exponential Moving Average of a sequence of angles.

    Parameters:
    - data: list or numpy array of data points (angles in radians)
    - alpha: float, the smoothing factor, between 0 and 1. Higher values give more weight to recent data.

    Returns:
    - numpy array containing the EMA of the provided data.
    """
    ema = np.zeros(len(data))

    ema[0] = data[0]  # Start the EMA with the first data point
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    if visualize:
        visualize_smooth(name, data, ema)

    return ema

def gaussian_smooth(names, radians_list, sigma, visualize):
    # Apply a handle outlier function to the radian_list
    smoothed_radians_list = []
    for i in range(len(radians_list)):
        radian_temp = radians_list[i]
        name_temp = names[i]
        smoothed_radians = gaussian_smooth_angles(name_temp, radian_temp, sigma, visualize)

        smoothed_radians_float_list = [float(i) for i in smoothed_radians.tolist()]

        smoothed_radians_list.append(smoothed_radians_float_list)

    return smoothed_radians_list


def gaussian_smooth_angles(name, angles, sigma, visualize):
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)

    # Convert to Cartesian coordinates
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)

    # Apply Gaussian filter
    x_smooth = gaussian_filter1d(x, sigma)
    y_smooth = gaussian_filter1d(y, sigma)

    # Convert back to angles
    smoothed_angles_rad = np.arctan2(y_smooth, x_smooth)

    if visualize:
        visualize_smooth(name, angles, smoothed_angles_rad)

    return smoothed_angles_rad

if __name__ == '__main__':
    # Example usage:
    numbers = [2.5, 5.0, 7.5, 10.0]
    normalized_numbers = normalize(numbers)
    print("Original numbers:", numbers)
    print("Normalized numbers:", normalized_numbers)

    float_list = [-2.5, 0.0, 5.0, 7.5, 10.0]

    # Scale the list to the range [0, 10]
    scaled_list = scale_to_range(float_list)
    print(scaled_list)
