import numpy as np
import math
import pandas as pd

from utils import clamp_matrix


def calculate_hand_orientation(index_2_squeeze, middle_2_squeeze, pinky_2_squeeze, ring_2_squeeze, thumb_2_squeeze):
    """
    Calculates the mean of five directional vectors and returns the yaw, pitch, and roll.

    Parameters:
    v1, v2, v3, v4, v5 (numpy.ndarray): The five directional vectors, each of shape (3,).

    Returns:
    tuple: The yaw, pitch, and roll of the mean directional vector.
    """
    # Calculate the mean vector
    mean_vector = (index_2_squeeze + middle_2_squeeze + pinky_2_squeeze + ring_2_squeeze + thumb_2_squeeze) / 5

    # Normalize the mean vector
    mean_vector = mean_vector / np.linalg.norm(mean_vector)

    # Calculate the yaw, pitch, and roll
    yaw = math.atan2(mean_vector[1], mean_vector[0])
    pitch = math.asin(mean_vector[2])
    roll = math.atan2(mean_vector[0], mean_vector[1])

    return yaw, pitch, roll


def get_hand_angle_list(directional_vecs):
    # ------------------------------------------- Left -------------------------------------------
    # Vector from left index 2
    left_index_2_squeeze = np.squeeze(directional_vecs[:, [6], :])

    # Vector from left middle 2
    left_middle_2_squeeze = np.squeeze(directional_vecs[:, [9], :])

    # Vector from left pinky 2
    left_pinky_2_squeeze = np.squeeze(directional_vecs[:, [12], :])

    # Vector from left ring 2
    left_ring_2_squeeze = np.squeeze(directional_vecs[:, [15], :])

    # Vector from left thumb 2
    left_thumb_2_squeeze = np.squeeze(directional_vecs[:, [18], :])

    left_yaw_radians_list = []
    left_pitch_radians_list = []
    left_roll_radians_list = []

    left_yaw_degrees_list = []
    left_pitch_degrees_list = []
    left_roll_degrees_list = []

    for i in range(directional_vecs.shape[0]):
        left_yaw_radians, left_pitch_radians, left_roll_radians = calculate_hand_orientation(
            left_index_2_squeeze[i], left_middle_2_squeeze[i], left_pinky_2_squeeze[i], left_ring_2_squeeze[i],
            left_thumb_2_squeeze[i])

        left_yaw_radians_list.append(left_yaw_radians)
        left_pitch_radians_list.append(left_pitch_radians)
        left_roll_radians_list.append(left_roll_radians)

    for radians in left_yaw_radians_list:
        left_yaw_degrees_list.append(math.degrees(radians))

    for radians in left_pitch_radians_list:
        left_pitch_degrees_list.append(math.degrees(radians))

    for radians in left_roll_radians_list:
        left_roll_degrees_list.append(math.degrees(radians))

    print "left_hand_yaw_radians = {}".format(left_yaw_radians_list)
    print "left_hand_pitch_radians = {}".format(left_pitch_radians_list)
    print "left_hand_roll_radians = {}".format(left_roll_radians_list)

    print "left_hand_yaw_degrees = {}".format(left_yaw_degrees_list)
    print "left_hand_pitch_degrees = {}".format(left_pitch_degrees_list)
    print "left_hand_roll_degrees = {}".format(left_roll_degrees_list)

    # ------------------------------------------- Right -------------------------------------------
    # Vector from right index 2
    right_index_2_squeeze = np.squeeze(directional_vecs[:, [23], :])

    # Vector from right middle 2
    right_middle_2_squeeze = np.squeeze(directional_vecs[:, [26], :])

    # Vector from right pinky 2
    right_pinky_2_squeeze = np.squeeze(directional_vecs[:, [29], :])

    # Vector from right ring 2
    right_ring_2_squeeze = np.squeeze(directional_vecs[:, [32], :])

    # Vector from right thumb 2
    right_thumb_2_squeeze = np.squeeze(directional_vecs[:, [35], :])

    right_yaw_radians_list = []
    right_pitch_radians_list = []
    right_roll_radians_list = []

    right_yaw_degrees_list = []
    right_pitch_degrees_list = []
    right_roll_degrees_list = []

    for i in range(directional_vecs.shape[0]):
        right_yaw_radians, right_pitch_radians, right_roll_radians = calculate_hand_orientation(
            right_index_2_squeeze[i], right_middle_2_squeeze[i], right_pinky_2_squeeze[i], right_ring_2_squeeze[i],
            right_thumb_2_squeeze[i])

        right_yaw_radians_list.append(right_yaw_radians)
        right_pitch_radians_list.append(right_pitch_radians)
        right_roll_radians_list.append(right_roll_radians)

    for radians in right_yaw_radians_list:
        right_yaw_degrees_list.append(math.degrees(radians))

    for radians in right_pitch_radians_list:
        right_pitch_degrees_list.append(math.degrees(radians))

    for radians in right_roll_radians_list:
        right_roll_degrees_list.append(math.degrees(radians))

    print "right_hand_yaw_radians = {}".format(right_yaw_radians_list)
    print "right_hand_pitch_radians = {}".format(right_pitch_radians_list)
    print "right_hand_roll_radians = {}".format(right_roll_radians_list)

    print "right_hand_yaw_degrees = {}".format(right_yaw_degrees_list)
    print "right_hand_pitch_degrees = {}".format(right_pitch_degrees_list)
    print "right_hand_roll_degrees = {}".format(right_roll_degrees_list)
    print "------------------------------------------------ Hand ------------------------------------------------"

    return left_yaw_radians_list, left_pitch_radians_list, left_roll_radians_list, right_yaw_radians_list, right_pitch_radians_list, right_roll_radians_list
    # return left_yaw_radians, left_pitch_radians, left_roll_radians


if __name__ == '__main__':
    # Test
    obj = pd.read_pickle("../generation_results/_3IOjpSGFqY_164_0_12.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # clamp the matrix
    # process if the element is greater than 1 or less than -1
    directional_vecs = clamp_matrix(directional_vecs)

    # Only test the first frame
    get_hand_angle_list(directional_vecs)
