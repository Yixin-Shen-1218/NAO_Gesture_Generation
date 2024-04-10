#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use angleInterpolation Method"""

import qi
import argparse
import sys
import time
import almath


def main(session):
    """
    This example uses the angleInterpolation method.
    """
    # Get the service ALMotion.

    motion_service  = session.service("ALMotion")

    motion_service.setStiffnesses("Head", 1.0)

    # # Example showing a single target angle for one joint
    # # Interpolates the head yaw to 1.0 radian in 1.0 second
    # names      = "HeadYaw"
    # angleLists = 50.0*almath.TO_RAD
    # timeLists  = 1.0
    # isAbsolute = True
    # motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    #
    # time.sleep(1.0)
    #
    # # Example showing a single trajectory for one joint
    # # Interpolates the head yaw to 1.0 radian and back to zero in 2.0 seconds
    # names      = "HeadYaw"
    # #              2 angles
    # angleLists = [30.0*almath.TO_RAD, 0.0]
    # #              2 times
    # timeLists  = [1.0, 2.0]
    # isAbsolute = True
    # motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    #
    # time.sleep(1.0)
    #
    # # Example showing multiple trajectories
    # names      = ["HeadYaw", "HeadPitch"]
    # angleLists = [30.0*almath.TO_RAD, 30.0*almath.TO_RAD]
    # timeLists  = [1.0, 1.2]
    # isAbsolute = True
    # motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    # Example showing multiple trajectories
    # Interpolates the head yaw to 1.0 radian and back to zero in 2.0 seconds
    # while interpolating HeadPitch up and down over a longer period.
    names  = ["HeadYaw","HeadPitch"]
    # Each joint can have lists of different lengths, but the number of
    # angles and the number of times must be the same for each joint.
    # Here, the second joint ("HeadPitch") has three angles, and
    # three corresponding times.
    angleLists  = [[50.0*almath.TO_RAD, 0.0],
                   [-30.0*almath.TO_RAD, 30.0*almath.TO_RAD, 0.0]]
    timeLists   = [[1.011111111111111111111111111111111111111111, 2.0], [ 1.0, 2.0, 3.0]]
    isAbsolute  = True
    motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    motion_service.setStiffnesses("Head", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.61.144.92",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)