import math

import numpy as np
import pandas as pd


def calculate_hand_orientation(hand_wrist_squeeze):
    yaw_radians_list = []
    pitch_radians_list = []
    roll_radians_list = []

    yaw_degrees_list = []
    pitch_degrees_list = []
    roll_degrees_list = []




    print "hand_yaw_degrees = {}".format(yaw_degrees_list)
    print "hand_pitch_degrees = {}".format(pitch_degrees_list)
    print "hand_roll_degrees = {}".format(roll_degrees_list)

    return yaw_radians_list, pitch_radians_list, roll_radians_list

import numpy as np

def mean_vector(vectors):
    """Calculate the mean of the provided vectors."""
    return np.mean(vectors, axis=0)

def calculate_yaw_pitch(vector):
    """Calculate yaw and pitch from the vector."""
    x, y, z = vector
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))
    # Roll is not defined for a single vector without a reference frame.
    return np.degrees(yaw), np.degrees(pitch)

# Example directional vectors
vectors = np.array([
    [1, 2, 3],  # Example vector for index finger
    [2, 3, 4],  # Example vector for middle finger
    [1, 1, 1],  # Example vector for pinky
    [2, 2, 2],  # Example vector for ring finger
    [3, 1, 2]   # Example vector for thumb
])

# Calculate the mean vector
mean_vec = mean_vector(vectors)

# Calculate yaw and pitch
yaw, pitch = calculate_yaw_pitch(mean_vec)

print(f"Mean Vector: {mean_vec}")
print(f"Yaw: {yaw} degrees, Pitch: {pitch} degrees")


def get_hand_angle_list(directional_vecs):
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


    left_yaw_radians, left_pitch_radians, left_roll_radians = calculate_hand_orientation(
        left_hand_wrist_squeeze)

    print "left_hand_yaw_radians = {}".format(left_yaw_radians)
    print "left_hand_pitch_radians = {}".format(left_pitch_radians)
    print "left_hand_roll_radians = {}".format(left_roll_radians)

    # Vector from right hand
    right_hand_wrist_squeeze = np.squeeze(directional_vecs[:, [21], :])

    right_yaw_radians, right_pitch_radians, right_roll_radians = calculate_hand_orientation(
        right_hand_wrist_squeeze)
    print "right_hand_yaw_radians = {}".format(right_yaw_radians)
    print "right_hand_pitch_radians = {}".format(right_pitch_radians)
    print "right_hand_roll_radians = {}".format(right_roll_radians)

    return left_yaw_radians, left_pitch_radians, left_roll_radians, right_yaw_radians, right_pitch_radians, right_roll_radians


if __name__ == '__main__':
    # Test
    obj = pd.read_pickle("../generation_results/o1rERZRFyqE_112_4_0.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # Only test the first frame
    get_hand_angle_list(directional_vecs)
