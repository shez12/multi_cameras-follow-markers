import rospy
import time
import numpy as np
from spatialmath import SE3
import json
import sys
import os


from maker_tracker import same_position
from aruco_data.id_info import IDInfoList, IDInfo
from maker_tracker import MarkerTracker
from my_utils.robot_utils import robot_move,robot_fk,robot_ee2marker
from my_utils.myRobotSaver import MyRobotSaver,read_movement,replay_movement


sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 
import ros_utils.myGripper



def record_demo(robot,robot_name,filename): 
    recorder = MyRobotSaver(robot,robot_name, filename, init_node=False)
    recorder.record_movement()


def replay_demo_with_adjust(robot,tracker:MarkerTracker,marker_id,filename):
    '''
    args:
        robot: class of robot controller
        tracker: class of MarkerTracker
        marker_id: int, the id of the marker
        filename: string, path to the file containing the recorded movement
        goal_pose_in_ee: SE3, the goal pose of the marker in ee frame before replay

    '''

    init_marker_pose_in_base = tracker.get_marker_position(marker_id) # in robot base frame
    positions, velocities,transformations = read_movement(filename)
    # every 20 movement as a group, adjust the robot pose
    transformation_sequence = replay_movement(robot, positions, velocities,transformations,move_to_start=False)
    if len(transformation_sequence) == 0:
        print("No movement")
        return
    for i in range(len(transformation_sequence)%20):
        # check if the marker pose is changing
        current_marker_pose_in_base = tracker.get_marker_position(marker_id) # in robot base frame
        if current_marker_pose_in_base is not None:
            if not same_position(current_marker_pose_in_base,init_marker_pose_in_base):
                print("correcting robot pose")
                print("current_marker_pose_in_base",current_marker_pose_in_base)
                print("init_marker_pose_in_base",init_marker_pose_in_base)
                # robot.step_in_ee(actions=temp_transformation,wait=True)
                init_marker_pose_in_ee = robot_ee2marker(robot,init_marker_pose_in_base)
                current_marker_pose_in_ee = robot_ee2marker(robot,current_marker_pose_in_base)
                input("Press Enter to continue...")
                apply_robot_move(robot,current_marker_pose_in_ee,init_marker_pose_in_ee)
            init_marker_pose_in_base = tracker.get_marker_position(marker_id)
            max_index = min(i*20+20,len(transformation_sequence))
            robot.step_in_ee(actions=transformation_sequence[i*20:max_index],wait=True)
        input("Press Enter to continue...")
    print("finish replay")
           




def save_goal_pose(robot:init_robot,tracker:MarkerTracker,num_id):
    '''
    goal pose is the pose of the marker when the robot is at the initial pose
    args: 
        MarkerTracker: class of MarkerTracker
        num_id: int, the id of the marker

    '''
    num_id = str(num_id)
    goal_pose = None
    goal_corner = None
    # # Record initial marker pose
    marker_pose = None
    while marker_pose is None:

        marker_pose = tracker.get_marker_position(num_id) # in robot base frame
        marker_pose = robot_ee2marker(robot,marker_pose) # should be in camera frame
        corner  = tracker.corner["camera1"][num_id] # default camera is camera1 !!!!!

        # marker_pose = robot_ee2marker(robot,marker_pose)
        print("Still waiting for marker")

    goal_pose = marker_pose.copy()
    goal_corner = corner.copy()
    goal_pose.printline()
    # print("goal_corner",goal_corner)
    init_joints = robot.get_joints()  # Save current joints
    filename = f"robot1_aruco_{num_id}.json"
    record_demo(robot, robot_name='robot1', filename=filename)  # Record demonstration
    robot.move_joints(init_joints, duration=0.5, wait=True)
    

    if not os.path.exists(filename):
        # if file not exist, then save the initial pose
        id_info = IDInfo(num_id, goal_pose,goal_corner,filename )
        id_info_list = IDInfoList()
        id_info_list.load_from_json("aruco_data/data.json")
        # delete the old id_info
        num_id = str(num_id)
        if num_id in id_info_list.data:
            del id_info_list.data[num_id]
        id_info_list.add_id_info(id_info)
        id_info_list.save_to_json("aruco_data/data.json")
        print("save initial pose to data.json")

def apply_robot_move(robot, marker_pose, goal_pose):
    move = SE3(marker_pose) * SE3(goal_pose).inv()
    no_move = SE3(np.eye(4))
    rotation_norm = np.linalg.norm(move.R - np.eye(3))
    translation_norm = np.linalg.norm(move.t)
    

    if rotation_norm > 0.1 and translation_norm > 0.010:
        robot_move(robot, move.t, move.R)
    elif rotation_norm > 0.1:
        robot_move(robot, no_move.t, move.R)
    elif translation_norm > 0.010:
        robot_move(robot, move.t, no_move.R)
    else:
        print("marker_pose",marker_pose)
        print("goal_pose",goal_pose)
        print("no move")


class MarkerAction:
    def __init__(self, robot, gripper,tracker):
        self.robot = robot
        self.gripper = gripper
        self.tracker = tracker
        self.init_joints = robot.get_joints()   
        # State variables
        self.maker_id = None
        self.MARKER_SIZE = None
        self.goal_pose = None
        self.goal_corner = None
        self.goal_move_file = None
        self.follow_mode = False
        self.stop_flag = False
        self.key = None
        


    def handle_keypress(self, key):
        if key == 'q':
            self.stop_flag = True
            return True

        if key == 'f':
            self.follow_mode = not self.follow_mode
            print("follow_mode", self.follow_mode)
            time.sleep(2)

        if key == 'c':
            self.set_goal_object()

        if key == 'r' and self.goal_move_file is not None:
            self.execute_replay_sequence()

        if key == 'b':
            self.robot.move_joints(self.init_joints, duration=1, wait=True)
            self.gripper.set_gripper(1000, 5)

        return False

    def set_goal_object(self):
        self.maker_id = str(input("Enter the number of the object id: "))
        id_info_list = IDInfoList()
        id_info_list.load_from_json("aruco_data/data.json")
        id_info = id_info_list.data[str(self.maker_id)]
        self.goal_pose = SE3(id_info['pose']) # in camera frame
        print("goal_pose",self.goal_pose)
        self.goal_corner = np.array(id_info['corner'])
        self.goal_move_file = id_info['move']
        print("goal_id is", self.maker_id)

    def execute_replay_sequence(self):
        self.follow_mode = False
        replay_demo_with_adjust(self.robot,self.tracker,self.maker_id,self.goal_move_file)

    def run(self,key,marker_pose):
        if not self.stop_flag:
            # Handle keypress separately

            if key is not None:
                if self.handle_keypress(key):
                    return
            if marker_pose is None:
                return
            marker_pose = robot_ee2marker(self.robot, marker_pose)
            if key == 'm' and marker_pose is not None and self.goal_pose is not None:
                apply_robot_move(self.robot,marker_pose, self.goal_pose)
                return

            if key == 'x':
                print("marker_pose",marker_pose)
                print("goal_pose",self.goal_pose)
                print("goal_corner",self.goal_corner)
                print("current corner",self.tracker.corner["camera1"][self.maker_id])


            if self.follow_mode and self.goal_pose is not None:
                apply_robot_move(self.robot,marker_pose, self.goal_pose)
        



if __name__ == "__main__":
    try:
        # rospy.init_node('dino_bot')
        gripper = ros_utils.myGripper.MyGripper()
        robot = init_robot("robot1")
        tracker = MarkerTracker(camera_names=["camera1","camera3"])
        tracker.start_tracking(show_images=True)
        time.sleep(3)
        Action = MarkerAction(robot, gripper,tracker)
        while True:
            key1 = input("Enter command: ")
            marker_pose = tracker.get_marker_position(Action.maker_id)

            Action.run(key1,marker_pose)
            print("key1",key1)
    except KeyboardInterrupt:
        tracker.stop()
    
    # # record the movement
    # rospy.init_node('dino_bot')
    # gripper = ros_utils.myGripper.MyGripper()
    # robot = init_robot("robot1")
    # tracker = MarkerTracker()
    # tracker.start_tracking(show_images=True)
    # time.sleep(3)
    # save_goal_pose(robot,tracker,0)

