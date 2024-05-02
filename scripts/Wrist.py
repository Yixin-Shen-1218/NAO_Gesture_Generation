import math

import numpy as np
import pandas as pd
from numpy import mean

from utils import clamp_matrix, scale_to_range

Left_wrist_yaw_stand_init = math.radians(8.0)
Right_wrist_yaw_stand_init = math.radians(8.0)

# print Left_wrist_yaw_stand_init
# print Right_wrist_yaw_stand_init

left_wrist_yaw_radian_range = [math.radians(-30), math.radians(30)]
# print left_wrist_yaw_radian_range

right_wrist_yaw_radian_range = [math.radians(-30), math.radians(30)]
# print right_wrist_yaw_radian_range

def cross_product(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def normalize(v):
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def calculate_wrist_orientation(wrist_palm_vectors):
    # Calculate average (mean) vector for Forward direction
    F = tuple(sum(v[i] for v in wrist_palm_vectors) / len(wrist_palm_vectors) for i in range(3))
    # Assuming the third vector (index 2) is the middle finger for the Up direction
    U = wrist_palm_vectors[2]
    # Calculate Sideways direction
    S = cross_product(F, U)
    # Recalculate Up to ensure orthogonality
    U = cross_product(S, F)

    # Normalize vectors
    F = normalize(F)
    S = normalize(S)
    U = normalize(U)

    # # Calculate angles
    # yaw = math.atan2(F[0], F[1])
    # pitch = math.asin(-F[2])
    # roll = math.atan2(U[2], S[2])

    # Calculate angles
    pitch = math.atan2(F[0], F[1])
    # yaw = math.asin(-F[2])
    yaw = math.asin(-F[2])
    roll = math.atan2(U[2], S[2])

    return yaw, pitch, roll

# def calculate_wrist_orientation(wrist_palm_vectors):
#     """
#     Calculate the yaw, pitch, and roll angles for the wrist based on the unit directional vectors of the five fingers.
#
#     Args:
#         finger_vectors (list): A list of 5 numpy arrays representing the unit directional vectors of the fingers.
#
#     Returns:
#         tuple: A tuple containing the yaw, pitch, and roll angles in radians.
#     """
#     # Calculate the average vector of the finger vectors
#     average_vector = sum(wrist_palm_vectors) / len(wrist_palm_vectors)
#
#     # Calculate the yaw angle
#     yaw = math.atan2(average_vector[1], average_vector[0])
#
#     # Calculate the pitch angle
#     pitch = math.asin(-average_vector[2])
#
#     # Calculate the roll angle
#     roll = 0.0  # Assuming no roll for the wrist
#
#     return yaw, pitch, roll


def get_wrist_angle_list(directional_vecs):
    # ------------------------------------------- Left -------------------------------------------
    # Vector from left wrist-left index 1
    left_wrist_left_index_1_squeeze = np.squeeze(directional_vecs[:, [5], :])

    # Vector from left wrist-left middle 1
    left_wrist_left_middle_1_squeeze = np.squeeze(directional_vecs[:, [8], :])

    # Vector from left wrist-left pinky 1
    left_wrist_left_pinky_1_squeeze = np.squeeze(directional_vecs[:, [11], :])

    # Vector from left wrist-left ring 1
    left_wrist_left_ring_1_squeeze = np.squeeze(directional_vecs[:, [14], :])

    # Vector from left wrist-left thumb 1
    left_wrist_left_thumb_1_squeeze = np.squeeze(directional_vecs[:, [17], :])

    left_yaw_radians_list = []
    left_pitch_radians_list = []
    left_roll_radians_list = []

    left_yaw_degrees_list = []
    left_pitch_degrees_list = []
    left_roll_degrees_list = []

    for i in range(directional_vecs.shape[0]):
        left_yaw_radians, left_pitch_radians, left_roll_radians = calculate_wrist_orientation(
            [left_wrist_left_index_1_squeeze[i], left_wrist_left_middle_1_squeeze[i],
             left_wrist_left_pinky_1_squeeze[i],
             left_wrist_left_ring_1_squeeze[i], left_wrist_left_thumb_1_squeeze[i]])

        left_yaw_radians_list.append(left_yaw_radians)
        left_pitch_radians_list.append(left_pitch_radians)
        left_roll_radians_list.append(left_roll_radians)

    # left_yaw_radians_list_scaled = scale_to_range(left_yaw_radians_list, target_min = left_wrist_yaw_radian_range[0], target_max = left_wrist_yaw_radian_range[1])
    # print "yaw_radians_list_scaled = {}".format(left_yaw_radians_list_scaled)
    # left_yaw_degrees_list_scaled = []
    # for radians in left_yaw_radians_list_scaled:
    #     left_yaw_degrees_list_scaled.append(math.degrees(radians))
    #
    # print "left_yaw_degrees_list_scaled = {}".format(left_yaw_degrees_list_scaled)

    # left_yaw_radians_list = [abs(x) for x in left_yaw_radians_list]
    left_yaw_radians_list = [- x for x in left_yaw_radians_list]

    for radians in left_yaw_radians_list:
        left_yaw_degrees_list.append(math.degrees(radians))

    for radians in left_pitch_radians_list:
        left_pitch_degrees_list.append(math.degrees(radians))

    for radians in left_roll_radians_list:
        left_roll_degrees_list.append(math.degrees(radians))

    print "left_wrist_yaw_radians = {}".format(left_yaw_radians_list)
    print "left_wrist_pitch_radians = {}".format(left_pitch_radians_list)
    print "left_wrist_roll_radians = {}".format(left_roll_radians_list)

    print "left_wrist_yaw_degrees = {}".format(left_yaw_degrees_list)
    print "left_wrist_pitch_degrees = {}".format(left_pitch_degrees_list)
    print "left_wrist_roll_degrees = {}".format(left_roll_degrees_list)

    # ------------------------------------------- Right -------------------------------------------

    # Vector from right wrist-right index 1
    right_wrist_right_index_1_squeeze = np.squeeze(directional_vecs[:, [22], :])

    # Vector from right wrist-right middle 1
    right_wrist_right_middle_1_squeeze = np.squeeze(directional_vecs[:, [25], :])

    # Vector from right wrist-right pinky 1
    right_wrist_right_pinky_1_squeeze = np.squeeze(directional_vecs[:, [28], :])

    # Vector from right wrist-right ring 1
    right_wrist_right_ring_1_squeeze = np.squeeze(directional_vecs[:, [31], :])

    # Vector from right wrist-right thumb 1
    right_wrist_right_thumb_1_squeeze = np.squeeze(directional_vecs[:, [34], :])

    right_yaw_radians_list = []
    right_pitch_radians_list = []
    right_roll_radians_list = []

    right_yaw_degrees_list = []
    right_pitch_degrees_list = []
    right_roll_degrees_list = []

    for i in range(directional_vecs.shape[0]):
        right_yaw_radians, right_pitch_radians, right_roll_radians = calculate_wrist_orientation(
            [right_wrist_right_index_1_squeeze[i], right_wrist_right_middle_1_squeeze[i],
             right_wrist_right_pinky_1_squeeze[i],
             right_wrist_right_ring_1_squeeze[i], right_wrist_right_thumb_1_squeeze[i]])

        right_yaw_radians_list.append(right_yaw_radians)
        right_pitch_radians_list.append(right_pitch_radians)
        right_roll_radians_list.append(right_roll_radians)

    # right_yaw_radians_list_scaled = scale_to_range(right_yaw_radians_list, target_min = right_wrist_yaw_radian_range[0], target_max = right_wrist_yaw_radian_range[1])
    # print "right_yaw_radians_list_scaled = {}".format(right_yaw_radians_list_scaled)
    # right_yaw_degrees_list_scaled = []
    # for radians in right_yaw_radians_list_scaled:
    #     right_yaw_degrees_list_scaled.append(math.degrees(radians))
    #
    # print "right_yaw_degrees_list_scaled = {}".format(right_yaw_degrees_list_scaled)

    right_yaw_radians_list = [abs(x) for x in right_yaw_radians_list]

    for radians in right_yaw_radians_list:
        right_yaw_degrees_list.append(math.degrees(radians))

    for radians in right_pitch_radians_list:
        right_pitch_degrees_list.append(math.degrees(radians))

    for radians in right_roll_radians_list:
        right_roll_degrees_list.append(math.degrees(radians))

    print "right_wrist_yaw_radians = {}".format(right_yaw_radians_list)
    print "right_wrist_pitch_radians = {}".format(right_pitch_radians_list)
    print "right_wrist_roll_radians = {}".format(right_roll_radians_list)

    print "right_wrist_yaw_degrees = {}".format(right_yaw_degrees_list)
    print "right_wrist_pitch_degrees = {}".format(right_pitch_degrees_list)
    print "right_wrist_roll_degrees = {}".format(right_roll_degrees_list)
    print "------------------------------------------------ Wrist ------------------------------------------------"

    # return left_yaw_radians_list_scaled, left_pitch_radians_list, left_roll_radians_list, right_yaw_radians_list_scaled, right_pitch_radians_list, right_roll_radians_list
    return left_yaw_radians_list, left_pitch_radians_list, left_roll_radians_list, right_yaw_radians_list, right_pitch_radians_list, right_roll_radians_list


if __name__ == '__main__':
    # Test
    obj = pd.read_pickle("../generation_results/HBpoEMD25Io_156_0_11.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # clamp the matrix
    # process if the element is greater than 1 or less than -1
    directional_vecs = clamp_matrix(directional_vecs)

    # Only test the first frame
    left_yaw_radians_list, left_pitch_radians_list, left_roll_radians_list, right_yaw_radians_list, right_pitch_radians_list, right_roll_radians_list = get_wrist_angle_list(directional_vecs)
    print "left_yaw_radians[0]", left_yaw_radians_list[0], "left_yaw_radians[0] degree", math.degrees(left_yaw_radians_list[0])
    print "left_yaw_radians[14]", left_yaw_radians_list[14], "left_yaw_radians[14] degree", math.degrees(left_yaw_radians_list[14])
    print "left_yaw_radians[69]", left_yaw_radians_list[69], "left_yaw_radians[69] degree", math.degrees(left_yaw_radians_list[69])

    print "right_yaw_radians[0]", right_yaw_radians_list[0], "right_yaw_radians[0] degree", math.degrees(right_yaw_radians_list[0])
    print "right_yaw_radians[14]", right_yaw_radians_list[14], "right_yaw_radians[14] degree", math.degrees(right_yaw_radians_list[14])
    print "right_yaw_radians[69]", right_yaw_radians_list[69], "right_yaw_radians[69] degree", math.degrees(right_yaw_radians_list[69])
