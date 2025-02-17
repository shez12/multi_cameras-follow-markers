import rospy
import time
import numpy as np
import sys
import collections
import dm_env
import cv2

from my_utils.record_utils import Recorder,ImageRecorder

sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 


class RealEnv:
    """
    Observation space: {"qpos": Concat[ arm (6),          # absolute joint position

                        "images": {"camera1": (480x640x3),        # h, w, c, dtype='uint8'
                                   "camera2": (480x640x3),         # h, w, c, dtype='uint8'
                                   ....}
    """

    def __init__(self, robot_names,camera_names = ['camera1']):
        try:
            rospy.init_node('ik_step', anonymous=True)
        except:
            pass
        self.robot_names = robot_names
        self.robots = {}
        self.robot_infos = {}
        for i, name in enumerate(robot_names):
            self.robots[name] = init_robot(name)
            self.robot_infos[name] = Recorder(name, init_node=False) 
        self.image_recorder = ImageRecorder(init_node=False,camera_names = camera_names)

        time.sleep(2)


    def get_qpos(self,robot_names):
        '''
        args:
            robot_names: list of str
        return:
            qpos: np.ndarray, shape (6,)
        '''
        qpos = []
        for robot_name in robot_names:
            robot_qpos_raw = self.robot_infos[robot_name].qpos
            qpos.append(robot_qpos_raw)

        return np.concatenate(qpos)

    def get_data(self,robot_names):
        data = []
        for robot_name in robot_names:
            data.append(self.robot_infos[robot_name].data)
        return data

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(self.robot_names)
        obs['images'] = self.get_images()
        return obs





if __name__ == "__main__":
    robot_names = ['robot1']
    camera_name = ['camera1','camera3']
    env = RealEnv(robot_names,camera_name)
    while True:
        obs = env.get_observation()
        print(obs['qpos'])
        
        # Display images from all cameras
        for cam_name, img in obs['images'].items():
            cv2.imshow(cam_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        time.sleep(1)



   