import rospy
from sensor_msgs.msg import Image,Imu

import pyrealsense2 as rs
from cv_bridge import CvBridge
import cv2
import time
import os
import torch
import sys
import numpy as np



class MyImageSaver:
    def __init__(self, cameraNS="camera1"):
        self.bridge = CvBridge()
        self.cameraNS = cameraNS
        self.rgb_image = None
        self.depth_image = None
        self.accel = None
        self.rgb_sub = rospy.Subscriber(f'/{cameraNS}/color/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber(f'/{cameraNS}/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.imu_sub = rospy.Subscriber(f'/{cameraNS}/accel/sample', Imu, self.imu_callback)
        self.count = 0
        self.folder_path = "data/images"+time.strftime("-%Y%m%d-%H%M%S")
        #wait to receive first image
        time.sleep(1)
        self.hole_filling = rs.hole_filling_filter()

        while self.rgb_image is None:
            rospy.sleep(0.1)
            print(self.rgb_image)
            print('waiting for first image')
        print(f'init MyImageSaver at {self.folder_path}')

    def rgb_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # show the image
        except Exception as e:
            rospy.logerr("Error saving RGB image: %s", str(e))

    def imu_callback(self, data):
        '''     
        '''
        try:
            self.accel = [data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]
            # print(self.accel)
        except Exception as e:
            rospy.logerr("Error saving RGB image: %s", str(e))


    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            # self.depth_image = cv2.GaussianBlur(self.depth_image, (5, 5), 0)
        except Exception as e:
            rospy.logerr("Error saving depth image: %s", str(e))

    def generate_timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def save_image(self, image, prefix):
        os.makedirs(self.folder_path, exist_ok=True)
        prefix = self.cameraNS+'_'+prefix
        image_filename = os.path.join(self.folder_path,f"{prefix}_{self.count}.jpg")
        cv2.imwrite(image_filename, image)
        print(f"write to {image_filename}")
        return image_filename

    def record(self):
        #if extractor is None:
        file_path1 = self.save_image(self.rgb_image, "rgb")
        file_path2 = self.save_image(self.depth_image, 'depth')
        self.count += 1
        return file_path1, file_path2


    def spin(self):
        rospy.spin()




#





if __name__ == '__main__':
    try:
        rospy.init_node('image_saver')
        import time
        image_saver = MyImageSaver("camera1")
        time.sleep(1)
        # Example usage: Save RGB and depth images
        while not rospy.is_shutdown():
            image_saver.record()  # Save images
            time.sleep(1)  # Sleep for 1 seconds
        image_saver.spin()

    except rospy.ROSInterruptException:
        pass
