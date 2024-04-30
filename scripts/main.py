import argparse
import copy
import math
import sys
import time

import almath
import numpy as np
import pandas as pd
import qi
from scipy.signal import savgol_filter

from Head import get_head_angle_list
from Shoulder import get_shoulder_angle_list
from Elbow import get_elbow_angle_list
from Wrist import get_wrist_angle_list
from Hand import get_hand_angle_list
from utils import clamp_matrix, normalize, get_timestamps, generate_n_copies_of_time_list, remove_redundant_frames, \
    handle_outlier, moving_average, savgol_filter_smooth_radian_list, \
    exponential_moving_average_smooth, gaussian_smooth


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

def play_sound(path, session):
    audio_player_service = session.service("ALAudioPlayer")

    # Loads a file and launches the playing the audio
    fileId = audio_player_service.loadFile(path)

    time.sleep(2)
    audio_player_service.play(fileId, _async=True)

def main(session, directional_vecs, path, visualize):
    play_sound(path, session)

    # Get the service ALMotion.
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # Example showing how to activate "Move", "LArm" and "RArm" external anti collision
    name = "All"
    enable = True
    motion_service.setExternalCollisionProtectionEnabled(name, enable)

    # Example showing how to activate "Arms" anticollision
    chainName = "Arms"
    enable = True
    motion_service.setCollisionProtectionEnabled(chainName, enable)

    # Wake up robot
    motion_service.wakeUp()
    motion_service.setStiffnesses("Body", 1.0)

    # Send robot to Pose Init
    posture_service.goToPosture("Stand", 1)

    # Example showing multiple trajectories
    names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "RShoulderPitch", "RShoulderRoll", "LElbowYaw",
             "LElbowRoll", "RElbowYaw", "RElbowRoll", "LWristYaw", "RWristYaw", "LHand", "RHand"]

    head_yaw_radians, head_pitch_radians, head_roll_radians = get_head_angle_list(directional_vecs)
    print "\n\n"

    left_shoulder_yaw_radians, left_shoulder_pitch_radians, left_shoulder_roll_radians, right_shoulder_yaw_radians, right_shoulder_pitch_radians, right_shoulder_roll_radians = get_shoulder_angle_list(
        directional_vecs)
    print "\n\n"

    left_elbow_yaw_radians, left_elbow_pitch_radians, left_elbow_roll_radians, right_elbow_yaw_radians, right_elbow_pitch_radians, right_elbow_roll_radians = get_elbow_angle_list(
        directional_vecs)
    print "\n\n"

    left_wrist_yaw_radians, left_wrist_pitch_radians, left_wrist_roll_radians, right_wrist_yaw_radians, right_wrist_pitch_radians, right_wrist_roll_radians = get_wrist_angle_list(
        directional_vecs)
    print "\n\n"

    left_hand_yaw_radians, left_hand_pitch_radians, left_hand_roll_radians, right_hand_yaw_radians, right_hand_pitch_radians, right_hand_roll_radians = get_hand_angle_list(
        directional_vecs)
    print "\n\n"

    # convert the to 0-1
    left_hand_roll_radians_normalized = normalize(left_hand_roll_radians)
    right_hand_roll_radians_normalized = normalize(right_hand_roll_radians)

    # Use the min-max interpolation

    frame_number = directional_vecs.shape[0]

    timeList = get_timestamps(15, frame_number)
    # print timeList

    # timeList = [x + 2 for x in timeList]
    # print timeList

    # -------------------------------- smoothing the angle list --------------------------------
    radians_list = [head_yaw_radians, head_pitch_radians, left_shoulder_pitch_radians,
                    left_shoulder_roll_radians, right_shoulder_pitch_radians, right_shoulder_roll_radians,
                    left_elbow_yaw_radians, left_elbow_roll_radians, right_elbow_yaw_radians,
                    right_elbow_roll_radians, left_wrist_yaw_radians, right_wrist_yaw_radians]
    print "before smoothing =", radians_list[7]

    smoothed_radians_list = copy.deepcopy(radians_list)
    # smoothed_radians_list = savgol_filter_smooth_radian_list(names, radians_list, window_size=41, polyorder=3, visualize=visualize)
    # smoothed_radians_list = handle_outlier(names, radians_list, threshold=2, visualize=visualize)
    # smoothed_radians_list = moving_average(names, radians_list, window_size=5, visualize=visualize)
    # smoothed_radians_list = exponential_moving_average_smooth(names, radians_list, alpha=0.3, visualize=visualize)
    # smoothed_radians_list = gaussian_smooth(names, radians_list, sigma=5, visualize=visualize)
    print "after smoothing =", smoothed_radians_list[7]
    smoothed_radians_list.insert(len(smoothed_radians_list), left_hand_roll_radians_normalized)
    smoothed_radians_list.insert(len(smoothed_radians_list), right_hand_roll_radians_normalized)
    # ------------------------------------------------------------------------------------------

    # -------------------------------- Other Testing --------------------------------
    # timeLists = generate_n_copies_of_time_list(timeList[:1], n=len(names))
    # isAbsolute = True
    #
    # motion_service.angleInterpolation(names, [head_yaw_radians[:1], head_pitch_radians[:1], left_shoulder_pitch_radians[:1],
    #                                           left_shoulder_roll_radians[:1],
    #                                           right_shoulder_pitch_radians[:1], right_shoulder_roll_radians[:1],
    #                                   left_elbow_yaw_radians[:1], left_elbow_roll_radians[:1], right_elbow_yaw_radians[:1],
    #                                   right_elbow_roll_radians[:1], left_wrist_yaw_radians[:1], right_wrist_yaw_radians[:1],
    #                                   left_hand_roll_radians_normalized[:1], right_hand_roll_radians_normalized[:1]],
    #                                   timeLists, isAbsolute)

    # timeLists = generate_n_copies_of_time_list(timeList[:15], n=len(names))
    # isAbsolute = True
    #
    # motion_service.angleInterpolation(names, [head_yaw_radians[:15], head_pitch_radians[:15], left_shoulder_pitch_radians[:15],
    #                                           left_shoulder_roll_radians[:15],
    #                                           right_shoulder_pitch_radians[:15], right_shoulder_roll_radians[:15],
    #                                   left_elbow_yaw_radians[:15], left_elbow_roll_radians[:15], right_elbow_yaw_radians[:15],
    #                                   right_elbow_roll_radians[:15], left_wrist_yaw_radians[:15], right_wrist_yaw_radians[:15],
    #                                   left_hand_roll_radians_normalized[:15], right_hand_roll_radians_normalized[:15]],
    #                                   timeLists, isAbsolute)

    # timeLists = generate_n_copies_of_time_list(timeList[:30], n=len(names))
    # isAbsolute = True
    #
    # motion_service.angleInterpolation(names, [head_yaw_radians[:30], head_pitch_radians[:30], left_shoulder_pitch_radians[:30],
    #                                           left_shoulder_roll_radians[:30],
    #                                           right_shoulder_pitch_radians[:30], right_shoulder_roll_radians[:30],
    #                                   left_elbow_yaw_radians[:30], left_elbow_roll_radians[:30], right_elbow_yaw_radians[:30],
    #                                   right_elbow_roll_radians[:30], left_wrist_yaw_radians[:30], right_wrist_yaw_radians[:30],
    #                                   left_hand_roll_radians_normalized[:30], right_hand_roll_radians_normalized[:30]],
    #                                   timeLists, isAbsolute)
    # ------------------------------------------------------------------------------------------

    timeLists = generate_n_copies_of_time_list(timeList, n=len(names))
    print len(timeLists)
    isAbsolute = True

    # motion_service.angleInterpolation(names, smoothed_radians_list, timeLists, isAbsolute)
    motion_service.angleInterpolationBezier(names, timeLists, smoothed_radians_list)
    # # Go to rest position
    motion_service.rest()

    # get_angles()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.61.144.206",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    parser.add_argument("--visualize", type=bool, default=True,
                        help="Visualize the smoothing result")

    args = parser.parse_args()

    file_name = "_soZahWpS-0_57_0_8"
    obj = pd.read_pickle("../generation_results/{}.pkl".format(file_name))
    path = "/data/home/nao/audio/{}.wav".format(file_name)

    directional_vecs = remove_redundant_frames(obj)

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
    main(session, directional_vecs, path, args.visualize)
