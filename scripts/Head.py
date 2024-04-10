import numpy as np
import almath  # Import almath for angle conversions
import pandas as pd


def calculate_head_orientation(neck_to_nose, nose_to_right_eye, nose_to_left_eye, right_eye_to_right_ear,
                               left_eye_to_left_ear):
    # Calculate yaw using the neck_to_nose vector
    # Assuming the vector points forward from the nose, the yaw can be the angle in the horizontal plane
    yaw = float(np.arctan2(neck_to_nose[1], neck_to_nose[0]))

    # Calculate pitch using the neck_to_nose vector
    # Assuming the neck_to_nose vector's Z component gives us the pitch information
    pitch = float(np.arcsin(neck_to_nose[2]))

    # Calculate roll by the angle between the eyes to ears vectors
    # This is a simplification and assumes the face is symmetrical
    right_vector = right_eye_to_right_ear + nose_to_right_eye  # From right ear to nose
    left_vector = left_eye_to_left_ear + nose_to_left_eye  # From left ear to nose
    roll = float(np.arctan2(np.cross(right_vector, left_vector)[2], np.dot(right_vector, left_vector)))

    return yaw, pitch, roll


def get_head_angle_list(directional_vecs):
    # Vector from neck to nose
    neck_nose_squeeze = np.squeeze(directional_vecs[:, [37], :])
    # print neck_nose_squeeze.shape

    # Vector from nose to right eye
    nose_right_eye_squeeze = np.squeeze(directional_vecs[:, [38], :])
    # print nose_right_eye_squeeze.shape

    # Vector from nose to left eye
    nose_left_eye_squeeze = np.squeeze(directional_vecs[:, [39], :])
    # print nose_left_eye_squeeze.shape

    # Vector from right eye to right ear
    right_eye_ear_squeeze = np.squeeze(directional_vecs[:, [40], :])
    # print right_eye_ear_squeeze.shape

    # Vector from left eye to left ear
    left_eye_ear_squeeze = np.squeeze(directional_vecs[:, [41], :])
    # print left_eye_ear_squeeze.shape

    yaw_radians_list = []
    pitch_radians_list = []
    roll_radians_list = []

    for i in range(directional_vecs.shape[0]):
        yaw_radians, pitch_radians, roll_radians = calculate_head_orientation(neck_nose_squeeze[i],
                                                                              nose_right_eye_squeeze[i],
                                                                              nose_left_eye_squeeze[i],
                                                                              right_eye_ear_squeeze[i],
                                                                              left_eye_ear_squeeze[i])
        yaw_radians_list.append(yaw_radians)
        pitch_radians_list.append(pitch_radians)
        roll_radians_list.append(roll_radians)

    yaw_radians_first_temp = yaw_radians_list[0]
    pitch_radians_first_temp = pitch_radians_list[0]
    roll_radians_first_temp = roll_radians_list[0]
    for i, yaw_radians in enumerate(yaw_radians_list):
        yaw_radians_list[i] = yaw_radians_list[i] - yaw_radians_first_temp

    for i, pitch_radians in enumerate(pitch_radians_list):
        pitch_radians_list[i] = pitch_radians_list[i] - pitch_radians_first_temp

    for i, roll_radians in enumerate(roll_radians_list):
        roll_radians_list[i] = roll_radians_list[i] - roll_radians_first_temp

    print "yaw_radians_list = {}".format(yaw_radians_list)
    print "pitch_radians_list = {}".format(pitch_radians_list)
    print "roll_radians_list = {}".format(roll_radians_list)

    return yaw_radians_list, pitch_radians_list, roll_radians_list


if __name__ == '__main__':
    # # TEST
    # # Define the unit directional vectors representing head segments
    # v1 = np.array([0.0, 0.0, 1.0])  # Vector from neck to nose
    # v2 = np.array([0.2, 0.3, 0.9])  # Vector from nose to right eye
    # v3 = np.array([-0.2, 0.3, 0.9])  # Vector from nose to left eye
    # v4 = np.array([0.5, 0.0, 0.8])  # Vector from right eye to right ear
    # v5 = np.array([-0.5, 0.0, 0.8])  # Vector from left eye to left ear
    #
    # # Calculate head orientation (yaw, pitch, roll) in radians
    # yaw_radians, pitch_radians, roll_radians = calculate_head_orientation(v1, v2, v3, v4, v5)
    #
    # # Print the calculated angles in radians
    # print("Head Yaw (Radians): {}".format(yaw_radians))
    # print("Head Pitch (Radians): {}".format(pitch_radians))
    # print("Head Roll (Radians): {}".format(roll_radians))

    # Test
    obj = pd.read_pickle("../generation_results/o1rERZRFyqE_112_4_0.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # Only test the first frame
    get_head_angle_list(directional_vecs)
