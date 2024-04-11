import math

import numpy as np
import pandas as pd

from utils import clamp_matrix

def calculate_elbow_orientation(elbow_wrist_squeeze):
    yaw_radians_list = []
    pitch_radians_list = []
    roll_radians_list = []

    yaw_degrees_list = []
    pitch_degrees_list = []
    roll_degrees_list = []

    for i, elbow_to_wrist_vector in enumerate(elbow_wrist_squeeze):
        # Calculate ElbowYaw (elbow's yaw angle)
        ElbowYaw = math.atan2(elbow_to_wrist_vector[1], elbow_to_wrist_vector[0])

        # Calculate ElbowPitch (elbow's pitch angle)
        ElbowPitch = math.asin(elbow_to_wrist_vector[2])

        # Calculate ElbowRoll (elbow's roll angle)
        ElbowRoll = math.atan2(elbow_to_wrist_vector[0], elbow_to_wrist_vector[1])

        # Convert angles from radians to degrees if needed
        ElbowYaw_deg = math.degrees(ElbowYaw)
        ElbowPitch_deg = math.degrees(ElbowPitch)
        ElbowRoll_deg = math.degrees(ElbowRoll)

        yaw_degrees_list.append(ElbowYaw_deg)
        pitch_degrees_list.append(ElbowPitch_deg)
        roll_degrees_list.append(ElbowRoll_deg)

        yaw_radians_list.append(ElbowYaw)
        pitch_radians_list.append(ElbowPitch)
        roll_radians_list.append(ElbowRoll)

    print "elbow_yaw_degrees = {}".format(yaw_degrees_list)
    print "elbow_pitch_degrees = {}".format(pitch_degrees_list)
    print "elbow_roll_degrees = {}".format(roll_degrees_list)

    return yaw_radians_list, pitch_radians_list, roll_radians_list


def get_elbow_angle_list(directional_vecs):
    # ------------------------------------------- Left -------------------------------------------
    # Vector from left elbow-wrist
    left_elbow_wrist_squeeze = np.squeeze(directional_vecs[:, [4], :])

    left_yaw_radians, left_pitch_radians, left_roll_radians = calculate_elbow_orientation(
        left_elbow_wrist_squeeze)

    print "left_elbow_yaw_radians = {}".format(left_yaw_radians)
    print "left_elbow_pitch_radians = {}".format(left_pitch_radians)
    print "left_elbow_roll_radians = {}".format(left_roll_radians)

    # ------------------------------------------- Right -------------------------------------------
    # Vector from right elbow-wrist
    right_elbow_wrist_squeeze = np.squeeze(directional_vecs[:, [21], :])

    right_yaw_radians, right_pitch_radians, right_roll_radians = calculate_elbow_orientation(
        right_elbow_wrist_squeeze)
    print "right_elbow_yaw_radians = {}".format(right_yaw_radians)
    print "right_elbow_pitch_radians = {}".format(right_pitch_radians)
    print "right_elbow_roll_radians = {}".format(right_roll_radians)

    return left_yaw_radians, left_pitch_radians, left_roll_radians, right_yaw_radians, right_pitch_radians, right_roll_radians


if __name__ == '__main__':
    # # Example directional vector from left shoulder to left elbow (already normalized)
    # shoulder_to_elbow_vector = [1.0, 0.0, 0.0]  # Replace with your actual normalized vector
    #
    # # Calculate shoulder angles
    # LShoulderPitch_rad, LShoulderRoll_rad, LShoulderYaw_rad = calculate_shoulder_orientation(shoulder_to_elbow_vector)
    #
    # print("LShoulderPitch: {} radians".format(LShoulderPitch_rad))
    # print("LShoulderRoll: {} radians".format(LShoulderRoll_rad))
    # print("LShoulderYaw: {} radians".format(LShoulderYaw_rad))

    # Test
    obj = pd.read_pickle("../generation_results/o1rERZRFyqE_112_4_0.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)
    # clamp the matrix
    # process if the element is greater than 1 or less than -1
    directional_vecs = clamp_matrix(directional_vecs)


    # Only test the first frame
    get_elbow_angle_list(directional_vecs)
