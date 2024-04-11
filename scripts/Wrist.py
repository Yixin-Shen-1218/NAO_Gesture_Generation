import math

import numpy as np
import pandas as pd

from utils import clamp_matrix

Left_wrist_yaw_stand_init =  math.radians(8.0)
Right_wrist_yaw_stand_init =  math.radians(7.2)

print Left_wrist_yaw_stand_init
print Right_wrist_yaw_stand_init

left_yaw_degrees_list = []
left_pitch_degrees_list = []
left_roll_degrees_list = []

right_yaw_degrees_list = []
right_pitch_degrees_list = []
right_roll_degrees_list = []

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

    # Calculate angles
    yaw = math.atan2(F[0], F[1])
    pitch = math.asin(-F[2])
    roll = math.atan2(U[2], S[2])

    return yaw, pitch, roll

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

    for i in range(directional_vecs.shape[0]):
        left_yaw_radians, left_pitch_radians, left_roll_radians = calculate_wrist_orientation([left_wrist_left_index_1_squeeze[i], left_wrist_left_middle_1_squeeze[i], left_wrist_left_pinky_1_squeeze[i],
    left_wrist_left_ring_1_squeeze[i], left_wrist_left_thumb_1_squeeze[i]])

        left_yaw_radians_list.append(left_yaw_radians)
        left_pitch_radians_list.append(left_pitch_radians)
        left_roll_radians_list.append(left_roll_radians)

    left_yaw_radians_first_temp = left_yaw_radians_list[0]
    left_pitch_radians_first_temp = left_pitch_radians_list[0]
    left_roll_radians_first_temp = left_roll_radians_list[0]
    for i, yaw_radians in enumerate(left_yaw_radians_list):
        left_yaw_radians_list[i] = left_yaw_radians_list[i] - left_yaw_radians_first_temp + Left_wrist_yaw_stand_init

    for i, pitch_radians in enumerate(left_pitch_radians_list):
        left_pitch_radians_list[i] = left_pitch_radians_list[i] - left_pitch_radians_first_temp

    for i, roll_radians in enumerate(left_roll_radians_list):
        left_roll_radians_list[i] = left_roll_radians_list[i] - left_roll_radians_first_temp


    print "left_yaw_radians_list = {}".format(left_yaw_radians_list)
    print "left_pitch_radians_list = {}".format(left_pitch_radians_list)
    print "left_roll_radians_list = {}".format(left_roll_radians_list)

    # print "left_wrist_yaw_degrees = {}".format(left_yaw_degrees_list)
    # print "left_wrist_pitch_degrees = {}".format(left_pitch_degrees_list)
    # print "left_wrist_roll_degrees = {}".format(left_roll_degrees_list)

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

    for i in range(directional_vecs.shape[0]):
        right_yaw_radians, right_pitch_radians, right_roll_radians = calculate_wrist_orientation([right_wrist_right_index_1_squeeze[i], right_wrist_right_middle_1_squeeze[i], right_wrist_right_pinky_1_squeeze[i],
    right_wrist_right_ring_1_squeeze[i], right_wrist_right_thumb_1_squeeze[i]])

        right_yaw_radians_list.append(right_yaw_radians)
        right_pitch_radians_list.append(right_pitch_radians)
        right_roll_radians_list.append(right_roll_radians)

    right_yaw_radians_first_temp = right_yaw_radians_list[0]
    right_pitch_radians_first_temp = right_pitch_radians_list[0]
    right_roll_radians_first_temp = right_roll_radians_list[0]
    for i, yaw_radians in enumerate(right_yaw_radians_list):
        right_yaw_radians_list[i] = right_yaw_radians_list[i] - right_yaw_radians_first_temp + Right_wrist_yaw_stand_init

    for i, pitch_radians in enumerate(right_pitch_radians_list):
        right_pitch_radians_list[i] = right_pitch_radians_list[i] - right_pitch_radians_first_temp

    for i, roll_radians in enumerate(right_roll_radians_list):
        right_roll_radians_list[i] = right_roll_radians_list[i] - right_roll_radians_first_temp


    print "right_yaw_radians_list = {}".format(right_yaw_radians_list)
    print "right_pitch_radians_list = {}".format(right_pitch_radians_list)
    print "right_roll_radians_list = {}".format(right_roll_radians_list)

    # print "right_wrist_yaw_degrees = {}".format(right_yaw_degrees_list)
    # print "right_wrist_pitch_degrees = {}".format(right_pitch_degrees_list)
    # print "right_wrist_roll_degrees = {}".format(right_roll_degrees_list)

    return left_yaw_radians_list, left_pitch_radians_list, left_roll_radians_list, right_yaw_radians_list, right_pitch_radians_list, right_roll_radians_list


if __name__ == '__main__':
    # Test
    obj = pd.read_pickle("../generation_results/o1rERZRFyqE_112_4_0.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # clamp the matrix
    # process if the element is greater than 1 or less than -1
    directional_vecs = clamp_matrix(directional_vecs)

    # Only test the first frame
    get_wrist_angle_list(directional_vecs)
