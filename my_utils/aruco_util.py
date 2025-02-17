import cv2
import cv2.aruco as aruco
import numpy as np
from my_utils.pose_util import Rt_to_pose,inverse_pose, pose_to_SE3
from my_utils.camera_intrinsics import intrinsics

# Dictionary of available ArUco dictionaries
ARUCO_DICTS = {
    '4x4_50': cv2.aruco.DICT_4X4_50,
    '4x4_100': cv2.aruco.DICT_4X4_100,
    '4x4_250': cv2.aruco.DICT_4X4_250,
    '5x5_50': cv2.aruco.DICT_5X5_50,
    '5x5_100': cv2.aruco.DICT_5X5_100,
    '5x5_250': cv2.aruco.DICT_5X5_250,
    '6x6_50': cv2.aruco.DICT_6X6_50,
    '6x6_100': cv2.aruco.DICT_6X6_100,
    '6x6_250': cv2.aruco.DICT_6X6_250,
    '7x7_50': cv2.aruco.DICT_7X7_50,
    '7x7_100': cv2.aruco.DICT_7X7_100,
    '7x7_250': cv2.aruco.DICT_7X7_250,
    'apriltag_16h5': cv2.aruco.DICT_APRILTAG_16h5,
    'apriltag_25h9': cv2.aruco.DICT_APRILTAG_25h9,
    'apriltag_36h10': cv2.aruco.DICT_APRILTAG_36h10,
    'apriltag_36h11': cv2.aruco.DICT_APRILTAG_36h11
}

# Default dictionary
ARUCO_DICT_NAME = ARUCO_DICTS['4x4_250']
my_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

def set_aruco_dict(dict_name):
    """
    Set the ArUco dictionary to use for detection and generation
    Args:
        dict_name (str): Name of the dictionary from ARUCO_DICTS
    Returns:
        my_aruco_dict: ArUco dictionary
    """
    global ARUCO_DICT_NAME, my_aruco_dict
    if dict_name in ARUCO_DICTS:
        ARUCO_DICT_NAME = ARUCO_DICTS[dict_name]
        my_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        return my_aruco_dict
    return None

# Function to detect ArUco markers
def detect_aruco(image, draw_flag=False, aruco_dict=None):
    """
    Detect ArUco markers in an image
    Args:
        image: Input image
        draw_flag: Whether to draw detected markers on the image
        aruco_dict: Optional custom ArUco dictionary. If None, uses default dictionary
    Returns:
        corners: List of detected marker corners
        ids: List of detected marker IDs
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use provided dictionary or fall back to default
    dict_to_use = aruco_dict if aruco_dict is not None else my_aruco_dict
    detector = aruco.ArucoDetector(dict_to_use, aruco.DetectorParameters())
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
    if corners is None or ids is None:
        return [], []
    else:
        # Refine corners to subpixel accuracy
        refined_corners = []
        for corner in corners:
            refined_corner = cv2.cornerSubPix(
                gray,  # Gray image
                corner.astype(np.float32),  # Initial corners
                winSize=(5, 5),  # Larger window size for better accuracy
                zeroZone=(-1, -1),  # Default: no dead zone
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)  # Higher precision and more iterations
            )
            refined_corners.append(refined_corner)
        
        # Draw detected markers on the image
        if draw_flag and ids is not None:
            image = aruco.drawDetectedMarkers(image, refined_corners, ids)  
        
        # Flatten the refined corners for compatibility
        refined_corners = [c.reshape(-1, 2) for c in refined_corners]
        ids = ids.flatten().tolist()
        return refined_corners, ids

# Function to generate ArUco markers
def generate_aruco_marker(marker_id, marker_size, output_file, aruco_dict=None):
    """
    Generate an ArUco marker image
    Args:
        marker_id: ID of the marker to generate
        marker_size: Size of the marker in pixels
        output_file: Path to save the marker image
        aruco_dict: Optional custom ArUco dictionary. If None, uses default dictionary
    """
    dict_to_use = aruco_dict if aruco_dict is not None else my_aruco_dict
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(dict_to_use, marker_id, marker_size, marker_image, 1)
    cv2.imwrite(output_file, marker_image)

def estimate_markers_poses(corners, marker_size, intrinsics,frame):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    '''
    # make sure the aruco's orientation in the camera view! 
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    mtx = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                    [0, intrinsics["fy"], intrinsics["cy"]],
                    [0, 0, 1]], dtype=np.float32)
    # distortion = np.zeros((5, 1))  # Assuming no distortion
    distortion  = np.array([[ 0.00377581 , 0.00568285 ,-0.00188039, -0.00102468 , 0.02337337]])

    poses = []
    for c in corners:
        ret, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion)
        if ret:
            tvec = tvec.reshape((3,))
            R, _ = cv2.Rodrigues(rvec)
            pose = Rt_to_pose(R, tvec)  # Ensure Rt_to_pose is correctly implemented
            poses.append(pose)
            cv2.drawFrameAxes(frame, mtx, distortion, rvec, tvec, marker_size)
        else:
            print("Pose estimation failed for one of the markers")
    return poses

def get_aruco_poses(corners, ids, intrinsics,frame,marker_size):
    # make sure the aruco's orientation in the camera view! 
    poses = estimate_markers_poses(corners, marker_size=marker_size, intrinsics=intrinsics,frame=frame)  # Marker size in meters
    poses_dict = {}
    # detected
    if ids is not None:
        for k, iden in enumerate(ids):
            poses_dict[iden]=poses[k] 
    return poses_dict

def get_cam_pose(frame, marker_size, intrinsics, aruco_dict=None):
    corners, ids = detect_aruco(frame, draw_flag=True, aruco_dict=aruco_dict)
    poses_dict = get_aruco_poses(corners=corners, ids=ids, intrinsics=intrinsics, frame=frame, marker_size=marker_size)
    id = 0
    current_cam = None
    if id in poses_dict:
        current_pose = poses_dict[id]
        current_cam = inverse_pose(current_pose)
    return current_cam

def get_marker_pose(frame, marker_size, id=0, draw=True, aruco_dict=None):
    id = int(id)
    corners, ids = detect_aruco(frame, draw_flag=draw, aruco_dict=aruco_dict)
    if ids is not None and len(ids)>0:
        poses_dict = get_aruco_poses(corners=corners, ids=ids, intrinsics=intrinsics, frame=frame, marker_size=marker_size)
        for i, c in zip(ids, corners):
            if i == id:
                pose = pose_to_SE3(poses_dict[id])
                return pose, c
    return None, None

    