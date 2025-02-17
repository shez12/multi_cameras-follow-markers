from my_utils.pose_util import Rt_to_pose, pose_to_SE3
from spatialmath import SE3
import numpy as np

def robot_move(robot,t_meters,R):
    """
    Inputs: t_meters: (x,y,z) translation in camera frame
            R: (3x3) array - rotation matrix in camera frame
            robot: class of robot controller
    
    Moves and rotates the robot according to the input translation and rotation.
    """
    transformations = SE3([
        [0.03391, -0.01653, 0.9993, 0.03622],
        [0.9994, -0.0091, -0.03407, -0.02485],
        [0.009689, 0.9998, 0.01621, -0.1034],
        [0, 0, 0, 1]
    ])
    # transformations = SE3.Ry(90,unit='deg') * SE3.Rz(90,unit='deg')
    pose = Rt_to_pose(R,t_meters)
    SE3_pose = pose_to_SE3(pose)
    SE3_pose = transformations * SE3_pose * transformations.inv()
 
 
    return robot.step_in_ee(actions=SE3_pose,wait=True)



def robot_segment_move(robot,t_meters,R):
    """
    Inputs: t_meters: (x,y,z) translation in camera frame
            R: (3x3) array - rotation matrix in camera frame
            robot: class of robot controller
    
    Moves and rotates the robot according to the input translation and rotation.
    """
    transformations = SE3([
        [0.03391, -0.01653, 0.9993, 0.03622],
        [0.9994, -0.0091, -0.03407, -0.02485],
        [0.009689, 0.9998, 0.01621, -0.1034],
        [0, 0, 0, 1]
    ])
    # transformations = SE3.Ry(90,unit='deg') * SE3.Rz(90,unit='deg')
    pose = Rt_to_pose(R,t_meters)
    SE3_pose = pose_to_SE3(pose)
    SE3_pose = transformations * SE3_pose * transformations.inv()
    current_joints = np.array(robot.get_joints())
    target_joints = np.array(robot.step_in_ee(actions=SE3_pose,wait=True,return_joints= True,should_move=False))
    joints_list = []
    for i in range(3):
        new_joints = current_joints+i/2*(target_joints-current_joints)
        joints_list.append(new_joints)
    return joints_list




            
def robot_fk(robot,pose):
    transformations = SE3([
        [0.03391, -0.01653, 0.9993, 0.03622],
        [0.9994, -0.0091, -0.03407, -0.02485],
        [0.009689, 0.9998, 0.01621, -0.1034],
        [0, 0, 0, 1]
    ])
    T_b2ee = robot.get_pose_se3()
    T_ee2marker = pose
    T_b2marker = T_b2ee * transformations* T_ee2marker
    return T_b2marker

def robot_ee2marker(robot,pose):
    '''
    Inputs: pose: (4x4) array - pose in base frame
            robot: class of robot controller
    
    Returns: (4x4) array - pose in end effector frame
    '''
    transformations = SE3([
        [0.03391, -0.01653, 0.9993, 0.03622],
        [0.9994, -0.0091, -0.03407, -0.02485],
        [0.009689, 0.9998, 0.01621, -0.1034],
        [0, 0, 0, 1]
    ])
    T_b2ee = robot.get_pose_se3()
    T_b2marker = pose
    T_ee2marker =transformations.inv() * T_b2ee.inv() * T_b2marker
    return T_ee2marker
