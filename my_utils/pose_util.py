from re import S
import numpy as np
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R

unit_vector = [1,0,0]

def normalize_vector(vector, l):
    # Calculate the length of the vector
    vector_length = np.linalg.norm(vector)
    # Normalize the vector
    normalized_vector = (vector / vector_length) * l
    return normalized_vector

def visualize_poses(poses, label="None", color='b', ax=None, autoscale=False):
    if ax is None:
        ax = plt.axes(projection ='3d')
        # Plot SLAM poses
        ax.set_autoscale_on(autoscale)

    poses = np.array(poses)
    if len(poses.shape)<2:
        poses = np.array([poses])

    # plot positions
    p = poses[:, :3]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=color, marker='o', label=label)

    if poses.shape[1]>3:
        # have orientation
        q = poses[:, 3:]
        # Plot orientation vectors
        for i in range(len(p)):
            # Convert quaternion to rotation matrix
            d = quat_to_R(q[i]) @ unit_vector
            d = normalize_vector(d, l=0.2)
            ax.quiver(p[i, 0], p[i, 1], p[i, 2], 
                      d[0], d[1], d[2], 
                      length=0.1, normalize=True, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Poses with Quaternion")
    plt.legend()
    return ax

def SE3_to_pose(Rt):
    return list(Rt.t) + list(R_to_quat(Rt.R))

def pose_to_SE3(pose):
    se3 = SE3.Tx(0)
    se3.R[:] = quat_to_R(pose[3:])
    se3.t = pose[:3]
    return se3


def xyzw2wxyz(q):
    return [q[3], q[0], q[1], q[2]]

def wxyz2xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def R_to_quat(R):
    import transforms3d.quaternions as tq
    # return wxyz2xyzw(tq.mat2quat(R))
    return Rotation.from_matrix(R).as_quat()

def quat_to_R(q):
    """
    quat: x,y,z,w
    """

    import transforms3d.quaternions as tq
    # return tq.quat2mat(xyzw2wxyz(q))
    return  Rotation.from_quat(q).as_matrix()

def pose_to_Rt(pose):
    R = quat_to_R(pose[3:])
    t = np.array(pose[:3])
    return R, t

def Rt_to_pose(R, t):
    pose = list(t) + list(R_to_quat(R))
    return np.array(pose)

def inverse_Rt(R, t):
    return R.T, -R.T @ t

def inverse_pose(pose):
    R, t = pose_to_Rt(pose)
    R_star, t_star = inverse_Rt(R, t)
    return Rt_to_pose(R_star, t_star)

def Rt_dot(R1, t1, R2, t2):
    #R2 = R1 dot R0
    R = np.dot(R1, R2)
    t = np.dot(R1, t2) + t1
    return R, t

def relative_rotation(q1, q2):
    # q1 to q2
    # Convert quaternions to rotation objects

    rot1 = quat_to_R(q1)
    rot2 = quat_to_R(q2)

    # Compute the relative rotation from rot1 to rot2
    relative_rotation = rot2 * rot1.inv()
    return relative_rotation.as_matrix()

def transform_pose(R, t, pose):
    pose_star = pose.copy()
    pose_star[:3] = np.dot(R, pose[:3])+t
    if len(pose)>3:
        pose_star[3:] =  R_to_quat(R @ quat_to_R(pose[3:]))
    return pose_star
 
def transform_poses(R, t, poses):
    # single vector
    poses = np.array(poses)
    if len(poses.shape)<2:
        return transform_pose(R, t, pose)
    transformed_poses = []
    for pose in poses:
        transformed_pose = transform_pose(R, t, pose)
        transformed_poses.append(transformed_pose)
    transformed_poses = np.vstack(transformed_poses)
    return transformed_poses

def append_vector(matrix, vector_to_append):
    assert len(matrix.shape) == 2, "Input matrix must be 2D"
    res = []
    for row in matrix:
        res.append(list(row)+list(vector_to_append))

    return np.array(res)

# make sure right handed coordintate
def find_transformation(X, Y):
    """
    from X to Y
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY

    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_rigid_transform(source_points, target_points, enable_R=True):
    # Ensure inputs are numpy arrays
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    # Compute centroids of each point set
    centroid_src = np.mean(source_points, axis=0)
    centroid_tgt = np.mean(target_points, axis=0)

    if enable_R:
        # Center the point clouds
        src_centered = source_points - centroid_src
        tgt_centered = target_points - centroid_tgt

        # Compute the covariance matrix
        H = np.dot(src_centered.T, tgt_centered)

        # Perform Singular Value Decomposition
        from scipy.linalg import svd
        U, _, Vt = svd(H)
        R = np.dot(Vt.T, U.T)
    else:
        R = np.eye(3)
    # Compute the translation vector
    t = centroid_tgt - np.dot(R, centroid_src)
    return R, t


def vec2mat(vec):
    if len(vec.shape)<2:
        vec = np.array([vec])   
    return vec

def poses_error(poses1, poses2):
    poses1 = vec2mat(poses1)
    poses2 = vec2mat(poses2)
    return np.mean(np.linalg.norm(poses1[:, :3] - poses2[:, :3], axis=1))


def se3_to_rpy(se3):
    """
    Convert an SE3 object to roll, pitch, and yaw angles.

    Parameters:
    se3 (SE3): The SE3 object from spatialmath.

    Returns:
    tuple: A tuple containing roll, pitch, and yaw angles in radians.
    """
    # Extract the rotation matrix from the SE3 object
    R = se3.R

    # Calculate roll, pitch, and yaw from the rotation matrix
    roll = np.arctan2(R[2, 1], R[2, 2])  # Rotation around x-axis
    pitch = np.arcsin(-R[2, 0])          # Rotation around y-axis
    yaw = np.arctan2(R[1, 0], R[0, 0])   # Rotation around z-axis

    return roll, pitch, yaw

def rpy_to_se3(roll, pitch, yaw):
    """
    Convert roll, pitch, and yaw angles to an SE3 object.

    Parameters:
    roll (float): Roll angle in radians.
    pitch (float): Pitch angle in radians.
    yaw (float): Yaw angle in radians.

    Returns:
    SE3: An SE3 object representing the rotation.
    """
    # Create rotation matrices for roll, pitch, and yaw

    roll = 0

    pitch = 0

    
    print("rpy",roll,pitch,yaw)

    se3 = SE3.Rz(yaw) * SE3.Ry(pitch) * SE3.Rx(roll)
    return se3




def matrix_smooth(matrix_list):
    """
    对输入的一组变换矩阵（SE3）进行平滑处理，去除异常值后返回平滑的平均矩阵。
    :param matrix_list: list of SE3 objects
    :return: 平滑后的平均 SE3 矩阵
    """
    if not matrix_list:
        raise ValueError("Input matrix list is empty.")
    
    # 提取平移向量和平移部分
    translations = np.array([se3.t for se3 in matrix_list])  # 每行是一个平移向量
    rotations = np.array([R.from_matrix(se3.R).as_quat() for se3 in matrix_list])  # 四元数表示旋转



    # 平均平移部分
    avg_translation = np.mean(translations, axis=0)

    # 平均旋转部分（加权四元数平均）
    avg_quaternion = quaternion_average(rotations)

    # 将平均结果构造为 SE3 矩阵
    avg_rotation = R.from_quat(avg_quaternion).as_matrix()
    
    output = np.eye(4)
    output[:3, :3] = avg_rotation
    output[:3, 3] = avg_translation

    return SE3(output)


def quaternion_average(quaternions):
    """
    使用四元数平均法。
    :param quaternions: Nx4 的四元数数组
    :return: 平均后的四元数
    """
    # 构建对称矩阵
    A = np.zeros((4, 4))
    for q in quaternions:
        q = q / np.linalg.norm(q)  # 确保四元数归一化
        A += np.outer(q, q)

    # 特征值分解，取最大特征值对应的特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    avg_quaternion = eigenvectors[:, np.argmax(eigenvalues)]
    return avg_quaternion / np.linalg.norm(avg_quaternion)  # 再次归一化




def compute_orientation_from_acceleration(acceleration):
    """
    Compute the orientation of a camera from linear acceleration.

    Parameters:
        acceleration (array-like): A 3D vector [ax, ay, az] representing linear acceleration in m/s^2.

    Returns:
        orientation (scipy.spatial.transform.Rotation): Orientation as a rotation object.
    """
    acceleration = np.array(acceleration)

    # Normalize acceleration to find gravity direction
    gravity = acceleration / np.linalg.norm(acceleration)

    # Assume gravity vector in world frame is [0, 0, -1]
    world_gravity = np.array([0, 0, -1])

    # Compute rotation axis and angle
    rotation_axis = np.cross(gravity, world_gravity)
    rotation_axis_magnitude = np.linalg.norm(rotation_axis)

    if rotation_axis_magnitude < 1e-6:  # If the vectors are nearly aligned
        return R.from_quat([0, 0, 0, 1])  # Identity quaternion

    rotation_axis = rotation_axis / rotation_axis_magnitude
    dot_product = np.dot(gravity, world_gravity)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Create rotation object
    orientation = R.from_rotvec(rotation_axis * rotation_angle)
    return orientation

if __name__=="__main__":
    folder = 'slam_data/0613-slam-aruco'
    joints_traj = np.load(f'{folder}/traj.npy')
    slam_poses = np.load(f'{folder}/slam_poses.npy')
    robot_poses = np.load(f'{folder}/robot_poses.npy')

    visualize_poses(robot_poses)
    plt.show()