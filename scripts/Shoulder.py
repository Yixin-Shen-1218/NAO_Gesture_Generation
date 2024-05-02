import math

import numpy as np
import pandas as pd

from utils import clamp_matrix, scale_to_range

def calculate_shoulder_orientation(shoulder_elbow_squeeze):
    yaw_radians_list = []
    pitch_radians_list = []
    roll_radians_list = []

    yaw_degrees_list = []
    pitch_degrees_list = []
    roll_degrees_list = []

    for i, shoulder_to_elbow_vector in enumerate(shoulder_elbow_squeeze):
        # Calculate ShoulderPitch (shoulder's pitch angle)
        ShoulderPitch = math.atan2(shoulder_to_elbow_vector[1], shoulder_to_elbow_vector[0])

        # Calculate ShoulderYaw (shoulder's yaw angle)
        ShoulderYaw = math.asin(shoulder_to_elbow_vector[2])

        # Calculate ShoulderRoll (shoulder's roll angle)
        ShoulderRoll = math.atan2(shoulder_to_elbow_vector[0], shoulder_to_elbow_vector[1])

        # Convert angles from radians to degrees if needed
        ShoulderYaw_deg = math.degrees(ShoulderYaw)
        ShoulderPitch_deg = math.degrees(ShoulderPitch)
        ShoulderRoll_deg = math.degrees(ShoulderRoll)

        yaw_degrees_list.append(ShoulderYaw_deg)
        pitch_degrees_list.append(ShoulderPitch_deg)
        roll_degrees_list.append(ShoulderRoll_deg)

        yaw_radians_list.append(ShoulderYaw)
        pitch_radians_list.append(ShoulderPitch)
        roll_radians_list.append(ShoulderRoll)

    print "yaw_degrees = {}".format(yaw_degrees_list)
    print "pitch_degrees = {}".format(pitch_degrees_list)
    print "roll_degrees = {}".format(roll_degrees_list)

    return yaw_radians_list, pitch_radians_list, roll_radians_list


def get_shoulder_angle_list(directional_vecs):
    # ------------------------------------------- Left -------------------------------------------
    # Vector from left shoulder-elbow
    left_shoulder_elbow_squeeze = np.squeeze(directional_vecs[:, [3], :])

    left_yaw_radians, left_pitch_radians, left_roll_radians = calculate_shoulder_orientation(
        left_shoulder_elbow_squeeze)

    print "left_shoulder_yaw_radians = {}".format(left_yaw_radians)
    print "left_shoulder_pitch_radians = {}".format(left_pitch_radians)
    print "left_shoulder_roll_radians = {}".format(left_roll_radians)

    # post-processing the right shoulder pitch
    left_pitch_radians_after_minus = [x + math.radians(0) for x in left_pitch_radians]
    left_pitch_degrees_after_minus = [math.degrees(x) for x in left_pitch_radians_after_minus]
    print "left_pitch_radians_after_minus = {}".format(left_pitch_radians_after_minus)
    print "left_pitch_degrees_after_minus = {}".format(left_pitch_degrees_after_minus)

    left_roll_radians_after_minus = [x + math.radians(0) for x in left_roll_radians]
    left_roll_degrees_after_minus = [math.degrees(x) for x in left_roll_radians_after_minus]
    print "left_roll_radians_after_minus = {}".format(left_roll_radians_after_minus)
    print "left_roll_degrees_after_minus = {}".format(left_roll_degrees_after_minus)
    # ------------------------------------------- Right -------------------------------------------
    # Vector from right shoulder-elbow
    right_shoulder_elbow_squeeze = np.squeeze(directional_vecs[:, [20], :])

    right_yaw_radians, right_pitch_radians, right_roll_radians = calculate_shoulder_orientation(
        right_shoulder_elbow_squeeze)

    print "right_shoulder_yaw_radians = {}".format(right_yaw_radians)
    print "right_shoulder_pitch_radians = {}".format(right_pitch_radians)
    print "right_shoulder_roll_radians = {}".format(right_roll_radians)

    # post-processing the right shoulder pitch
    right_pitch_radians_after_minus = [math.radians(180) - x for x in right_pitch_radians]
    right_pitch_degrees_after_minus = [math.degrees(x) for x in right_pitch_radians_after_minus]
    print "right_pitch_radians_after_minus = {}".format(right_pitch_radians_after_minus)
    print "right_pitch_degrees_after_minus = {}".format(right_pitch_degrees_after_minus)

    print "------------------------------------------------ Shoulder ------------------------------------------------"

    return left_yaw_radians, left_pitch_radians_after_minus, left_roll_radians_after_minus, right_yaw_radians, right_pitch_radians_after_minus, right_roll_radians


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
    obj = pd.read_pickle("../generation_results/_soZahWpS-0_57_0_8.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # clamp the matrix
    # process if the element is greater than 1 or less than -1
    directional_vecs = clamp_matrix(directional_vecs)
    # print directional_vecs.shape

    # Only test the first frame
    left_yaw_radians, left_pitch_radians, left_roll_radians, right_yaw_radians, right_pitch_radians_after_minus, right_roll_radians = get_shoulder_angle_list(directional_vecs)
    print "left_pitch_radians[0]", left_pitch_radians[0], "left_pitch_radians[0] degree", math.degrees(left_pitch_radians[0])
    print "left_pitch_radians[15]", left_pitch_radians[15], "left_pitch_radians[15] degree", math.degrees(left_pitch_radians[15])
    print "left_roll_radians[0]", left_roll_radians[0], "left_roll_radians[0] degree", math.degrees(left_roll_radians[0])
    print "left_roll_radians[15]", left_roll_radians[15], "left_roll_radians[15] degree", math.degrees(left_roll_radians[15])

    print "right_pitch_radians_after_minus[0]", right_pitch_radians_after_minus[0], "right_pitch_radians_after_minus[0] degree", math.degrees(right_pitch_radians_after_minus[0])
    print "right_pitch_radians_after_minus[15]", right_pitch_radians_after_minus[15], "right_pitch_radians_after_minus[15] degree", math.degrees(right_pitch_radians_after_minus[15])
    print "right_roll_radians[0]", right_roll_radians[0], "right_roll_radians[0] degree", math.degrees(right_roll_radians[0])
    print "right_roll_radians[15]", right_roll_radians[15], "right_roll_radians[15] degree", math.degrees(right_roll_radians[15])

