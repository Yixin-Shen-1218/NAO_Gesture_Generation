import argparse
import math
import sys
import time

import almath
import numpy as np
import pandas as pd
import qi

from Head import get_head_angle_list
from Shoulder import get_shoulder_angle_list
from Elbow import get_elbow_angle_list
from utils import clamp_matrix


# time list, the interval is 1
# timeLists  = range(1, yaw_radians_list + 1)

# time list, the interval is 0.5
# timeLists  = [i * 0.5 for i in range(1, 244 + 1)]

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


def get_angles():
    motion_service = session.service("ALMotion")

    # Example that finds the difference between the command and sensed angles.
    names = "LShoulderPitch"
    useSensors = True
    commandAngles = motion_service.getAngles(names, useSensors)
    print "Command angles:"
    print str(commandAngles)
    print ""

    # Example that finds the difference between the command and sensed angles.
    names = "LShoulderRoll"
    useSensors = True
    commandAngles = motion_service.getAngles(names, useSensors)
    print "Command angles:"
    print str(commandAngles)
    print ""

    # Example that finds the difference between the command and sensed angles.
    names = "RShoulderPitch"
    useSensors = True
    commandAngles = motion_service.getAngles(names, useSensors)
    print "Command angles:"
    print str(commandAngles)
    print ""

    # Example that finds the difference between the command and sensed angles.
    names = "RShoulderRoll"
    useSensors = True
    commandAngles = motion_service.getAngles(names, useSensors)
    print "Command angles:"
    print str(commandAngles)
    print ""


def main(session, directional_vecs):
    """
    This example uses the angleInterpolation method.
    """
    # Get the service ALMotion.

    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # Wake up robot
    motion_service.wakeUp()

    motion_service.setStiffnesses("Body", 1.0)

    # Send robot to Pose Init
    posture_service.goToPosture("Sit", 1)

    # Example showing multiple trajectories
    names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "RShoulderPitch", "RShoulderRoll", "LElbowYaw", "LElbowRoll",
             "RElbowYaw", "RElbowRoll"]

    head_yaw_radians, head_pitch_radians, head_roll_radians = get_head_angle_list(directional_vecs)
    print "\n\n"

    left_shoulder_yaw_radians, left_shoulder_pitch_radians, left_shoulder_roll_radians, right_shoulder_yaw_radians, right_shoulder_pitch_radians, right_shoulder_roll_radians = get_shoulder_angle_list(directional_vecs)
    print "\n\n"

    left_elbow_yaw_radians, left_elbow_pitch_radians, left_elbow_roll_radians, right_elbow_yaw_radians, right_elbow_pitch_radians, right_elbow_roll_radians = get_elbow_angle_list(directional_vecs)
    print "\n\n"

    frame_number = directional_vecs.shape[0]

    timeList = get_timestamps(1, frame_number)
    print timeList

    timeLists = generate_n_copies_of_time_list(timeList[:3], n=len(names))
    print len(timeLists)
    # print len(timeLists)
    # print timeLists
    isAbsolute = True

    motion_service.angleInterpolation(names, [head_yaw_radians[:3], head_pitch_radians[:3], left_shoulder_pitch_radians[:3],
                                              left_shoulder_roll_radians[:3],
                                              right_shoulder_pitch_radians[:3], right_shoulder_roll_radians[:3],
                                      left_elbow_yaw_radians[:3], left_elbow_roll_radians[:3], right_elbow_yaw_radians[:3],
                                      right_elbow_roll_radians[:3]],
                                      timeLists, isAbsolute)

    # timeLists = generate_n_copies_of_time_list(timeList, n=len(names))
    # print len(timeLists)
    # # print len(timeLists)
    # # print timeLists
    # isAbsolute = True
    # motion_service.angleInterpolation(names, [head_yaw_radians, head_pitch_radians, shoulder_left_pitch_radians,
    #                                           shoulder_left_roll_radians,
    #                                           shoulder_right_pitch_radians, shoulder_right_roll_radians],
    #                                   timeLists, isAbsolute)


    # # Go to rest position
    # motion_service.rest()

    # get_angles()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.61.144.92",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    obj = pd.read_pickle("../generation_results/o1rERZRFyqE_112_4_0.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)

    # clamp the matrix
    # process if the element is greater than 1 or less than -1
    directional_vecs = clamp_matrix(directional_vecs)


    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                                                                                              "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session, directional_vecs)
