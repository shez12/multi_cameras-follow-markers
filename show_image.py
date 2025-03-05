import rospy
import time
import numpy as np
from spatialmath import SE3
import json
import sys
import os
import argparse


from marker_tracker import same_position
from aruco_data.id_info import IDInfoList, IDInfo
from marker_tracker import MarkerTracker
from my_utils.aruco_util import get_marker_pose,set_aruco_dict
from my_utils.robot_utils import robot_move,robot_fk,robot_ee2marker
from my_utils.myRobotSaver import MyRobotSaver,read_movement,replay_movement
from follow_aruco import *
from marker_tracker import *






if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dino Bot Control Script')
    parser.add_argument('--camera1', type=str, default="camera1", help='Name of first camera')
    parser.add_argument('--camera2', type=str, default="camera3", help='Name of second camera')
    parser.add_argument('--robot', type=str, default="robot1", help='Robot name')
    args = parser.parse_args()

    rospy.init_node('dino_bot')
    gripper = ros_utils.myGripper.MyGripper()
    robot = init_robot(args.robot)
    tracker = MarkerTracker(camera_names=[args.camera1, args.camera2])
    tracker.start_tracking(show_images=True)
    time.sleep(3)
    Action = MarkerAction(robot, gripper, tracker)
    while True:
        key1 = tracker.show_image(args.camera1, Action.maker_id, Action.goal_corner)
        key2 = tracker.show_image(args.camera2, Action.maker_id, Action.goal_corner)

        if key1 == ord('c') or key2 == ord('c'):
            Action.set_goal_object()