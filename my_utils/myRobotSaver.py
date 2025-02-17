import rospy
from sensor_msgs.msg import JointState
import json
import os
import sys
import threading
import select
import termios
import tty

from my_utils.pose_util import *
sys.path.append("/home/hanglok/work/ur_slam")
from ik_step import init_robot


class MyRobotSaver:
    def __init__(self,robot,robot_name, filename, init_node=True):
        '''
        Record robot state
        '''


        self.robot = robot
        self.robot_name = robot_name
        self.filename = filename
        self.rate = rospy.Rate(10)  # 10 Hz
        self.qpos = None
        self.exit_flag = False  # Flag to control the recording loop

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        Subscriber_name = '/' + self.robot_name + '/joint_states'
        self.joint_state_sub = rospy.Subscriber(Subscriber_name, JointState, self.robot_state_cb)
        
        # Wait for self.qpos to be updated
        rospy.sleep(1)

        while self.qpos is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            print('waiting for qpos')   


        # Start the keyboard listener thread
        self.listener_thread = threading.Thread(target=self.listen_for_exit)
        self.listener_thread.daemon = True  # Daemonize thread
        self.listener_thread.start()

    def listen_for_exit(self):
        '''
        Listen for the 'q' key press to signal exit.
        '''
        # Save the terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # Set terminal to raw mode
            tty.setcbreak(fd)
            while not self.exit_flag and not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 'q':
                        rospy.loginfo("Detected 'q' key press. Exiting recording loop...")
                        self.exit_flag = True
                        break
        finally:
            # Restore the terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def robot_state_cb(self, data):
        position = list(data.position)
        velocity = list(data.velocity)
        position[0], position[2] = position[2], position[0]
        velocity[0], velocity[2] = velocity[2], velocity[0]
        self.qpos = position
        self.qvel = velocity
    def record_movement(self):
        '''
        Record robot movement; the recording stops when 'q' is pressed or the node is shut down.
        '''
        rospy.loginfo("Recording started... turn on robot control and Press 'q' to stop recording.")
        self.joint_positions = []
        self.joint_velocities = []
        self.tcp_transformations = []
        while not rospy.is_shutdown() and not self.exit_flag:
            self.joint_positions.append(self.qpos)
            self.joint_velocities.append(self.qvel)
            self.tcp_transformations.append(self.robot.myIK.fk(self.qpos))
            self.rate.sleep()
        rospy.loginfo("Exiting recording loop.")
        self.save_recording(self.filename)

    def save_recording(self, filename):
        # Create the directory if it doesn't exist
        if not os.path.exists('records'):
            os.makedirs('records')
        with open('records/' + filename, 'w') as f:
            # Save as dictionary
            json.dump({'positions': self.joint_positions, 'velocities': self.joint_velocities, 'tcp_transformations': self.tcp_transformations}, f)
        rospy.loginfo(f"Movement recorded and saved to {filename}.")
        print("Recording finished.")



def read_movement(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['positions'], data['velocities'], data['tcp_transformations']


def replay_movement(robot, positions, velocities,transformations, move_to_start=False):
    '''
    Replay the movement
    '''
    if move_to_start:
        robot.move_joints(positions[0], duration=0.5, wait=True)

    # current_joint_positions = robot.get_joints()
    move_transformation = []
    for i in range(len(transformations)-1):
        current_transformation = pose_to_SE3(transformations[i]).inv() @ pose_to_SE3(transformations[i+1])
        change_magnitude = np.linalg.norm(current_transformation.log())  # 使用对数映射计算位姿变化
        print(f"Change magnitude: {change_magnitude}")
        if change_magnitude > 0.001:
            move_transformation.append(current_transformation)
    
    print(f"Total {len(move_transformation)} movements")
    # input("Press Enter to replay the movement...")
    # remove the continuous ,same transformation
    rospy.loginfo(f"Replaying relative movement ....")

    return move_transformation


        
# Record the movement
if __name__ == '__main__':
    robot = init_robot("robot1")
    rospy.init_node('robot1_recorder', anonymous=False)
    recorder = MyRobotSaver(robot, 'robot1', filename='robot1_aruco_0.json', init_node=False)
    recorder.record_movement()
    input("Press Enter to replay the movement...")

# ## Replay the movement
#     rospy.init_node('robot1_player', anonymous=False)
    # positions, velocities ,transformations = read_movement('records/robot1_movements.json')
    # replay_movement(recorder.robot, positions, velocities, transformations, move_to_start=True)