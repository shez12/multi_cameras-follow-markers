import json
from spatialmath import SE3
import numpy as np

class IDInfo:
    def __init__(self, id, pose, corner, move):
        '''
        id: int
        pose: marker's goal pose
        move: list of movement after reach goal pose
        '''
        self.id = id
        self.pose = pose
        self.corner = corner
        self.move = move

    def to_dict(self):
        """Convert the IDInfo instance to a dictionary."""
        return {
            "pose": self.pose.A.tolist(),
            "corner": np.array(self.corner).tolist(),
            "move": self.move
        }

    def __str__(self):
        return f"ID: {self.id}, Pose: {self.pose}, Move: {self.move}"

class IDInfoList:
    def __init__(self):
        self.data = {}

    def add_id_info(self, id_info):
        self.data[id_info.id] = id_info.to_dict()

    def save_to_json(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def load_from_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

if __name__ == "__main__":
    num_id = "123"
    id_info_list = IDInfoList()
    id_info_list.load_from_json("/home/hanglok/work/hand_pose/aruco_data/data.json")
    # delete the old id_info
    if num_id in id_info_list.data:
        del id_info_list.data[num_id]
        print("delete id_info")
    id_info_list.save_to_json("/home/hanglok/work/hand_pose/aruco_data/data.json")
    print("save id_info")
