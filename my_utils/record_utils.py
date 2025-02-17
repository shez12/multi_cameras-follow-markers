'''
modified from https://github.com/tonyzhaozh/aloha

'''
import numpy as np
import time
import rospy
from collections import deque
from sensor_msgs.msg import Image
from control_msgs.msg import JointTrajectoryControllerState
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time
import h5py
import numpy as np
import cv2  # Add this import

import IPython
e = IPython.embed

class ImageRecorder:
    def __init__(self, init_node=True, camera_names=['camera1', 'camera2', 'usb_cam'], is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = camera_names
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'camera1':
                callback_func = self.image_cb_cam_1
            elif cam_name == 'camera2':
                callback_func = self.image_cb_cam_2
            elif cam_name == 'camera3':
                callback_func = self.image_cb_cam_3           
            elif cam_name == "usb_cam":
                callback_func = self.image_cb_usb_cam
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}/color/image_raw", Image, callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        setattr(self, f'{cam_name}_image', cv_image)
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)
            self.print_diagnostics()
            
            # Display the image
            cv2.imshow(f'{cam_name} Debug View', cv_image)
            cv2.waitKey(1)  # Refresh the window

    def image_cb_cam_1(self, data):
        cam_name = 'camera1'
        return self.image_cb(cam_name, data)

    def image_cb_cam_2(self, data):
        cam_name = 'camera2'
        return self.image_cb(cam_name, data)
    
    def image_cb_cam_3(self, data):
        cam_name = 'camera3'
        return self.image_cb(cam_name, data)
    
    def image_cb_usb_cam(self, data):
        cam_name = 'usb_cam'
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()

    def __del__(self):
        if self.is_debug:
            cv2.destroyAllWindows()

class Recorder:
    def __init__(self, robot_name, init_node=True,is_debug=False):
        '''
        Record robot state
        
    
        '''
        self.robot_name = robot_name
        self.secs = None
        self.nsecs = None
        self.qpos = None

        self.is_debug = is_debug


        if init_node:
            rospy.init_node('recorder', anonymous=True)
        Subscriber_name ='/'+ self.robot_name + '/scaled_pos_joint_traj_controller/state'
        self.joint_state_sub = rospy.Subscriber(Subscriber_name, JointTrajectoryControllerState, self.robot_state_cb)


        time.sleep(0.5)

    def robot_state_cb(self, data):
        self.qpos = data.actual.positions
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())


    def stop_subscriptions(self):
        self.joint_state_sub.unregister()
        # self.gripper_command_sub.unregister()


if __name__ == '__main__':
    recorder = ImageRecorder(camera_names=["usb_cam"],is_debug=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down recorder node.")
        recorder.stop_subscriptions()
