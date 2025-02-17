'''
### Acknowledgements
This project uses code or is inspired by [Pose-Tracking-for-Aruco-Markers-using-Kalman-Filter]
(https://github.com/SihanWang-WHU/Pose-Tracking-for-Aruco-Markers-using-Kalman-Filter)

'''

from gettext import translation
from statistics import variance
import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
from spatialmath import SE3
import threading
import queue


from my_utils.pose_util import SE3_to_pose,pose_to_SE3


class KalmenFilter:
    def __init__(self):
        self.epoch = 0
        self.dtime = 1 / 50
        covariance_x = 0.5
        covariance_z = 0.8
        self.H = np.empty([7, 14], dtype=float)
        self.A = np.empty([14, 14], dtype=float)
        self.xt = np.empty([14, 1], dtype=float)
        self.Pt = np.eye(14, dtype=float)
        self.zt = np.empty([7, 1], dtype=float)
        self.omegat = covariance_x * np.eye(14, dtype=float)
        self.vt = covariance_z * np.eye(7, dtype=float)
        self.x_pre = np.empty([14, 1], dtype=float)
        self.P_pre = np.eye(14, dtype=float)
    

    def new_markerpose(self,marker_pose):
        '''
        marker_pose in base frame: SE3 object 
        '''
        self.epoch += 1
        output = SE3_to_pose(marker_pose)
        self.transform_translation_x = output[0]
        self.transform_translation_y = output[1]
        self.transform_translation_z = output[2]
        self.transform_rotation_x = output[3]
        self.transform_rotation_y = output[4]
        self.transform_rotation_z = output[5]
        self.transform_rotation_w = output[6]
    

    def Kalman_Filter(self):
        # 利用Kalman滤波来推算位置/量测更新
        # xˆk/k−1 = Φk/k−1xˆk−1
        # Pk/k−1 = Φk/k−1Pk−1ΦT k/k−1 + Γk−1Qk−1ΓT k−1
        # 构造H矩阵
        H_left = np.eye(7, dtype=float)
        H_right = np.zeros([7, 7], dtype=float)
        self.H = np.hstack((H_left, H_right))
        # 构造A矩阵
        A_eye = np.eye(7, dtype=float)
        A_dtime = self.dtime * np.eye(7, dtype=float)
        A_up = np.hstack((A_eye, A_dtime))
        A_zeros = np.zeros([7, 7], dtype=float)
        A_down = np.hstack((A_zeros, A_eye))
        self.A = np.vstack((A_up, A_down))

        if (self.epoch == 1):
            # 第一个历元只做一步预测
            self.xt = np.array([[self.transform_translation_x], [self.transform_translation_y],[self.transform_translation_z],
                       [self.transform_rotation_x], [self.transform_rotation_y], [self.transform_rotation_z],
                       [self.transform_rotation_w], [0], [0], [0], [0], [0], [0], [0]])
            self.Predict()
            # 赋值
            self.stat_transform_translation_x = self.x_pre[0, 0]
            self.stat_transform_translation_y = self.x_pre[1, 0]
            self.stat_transform_translation_z = self.x_pre[2, 0]
            self.stat_transform_rotation_x = self.x_pre[3, 0]
            self.stat_transform_rotation_y = self.x_pre[4, 0]
            self.stat_transform_rotation_z = self.x_pre[5, 0]
            self.stat_transform_rotation_w = self.x_pre[6, 0]
            self.stat_transform_translation_x_dif = self.x_pre[7, 0]
            self.stat_transform_translation_y_dif = self.x_pre[8, 0]
            self.stat_transform_translation_z_dif = self.x_pre[9, 0]
            self.stat_transform_rotation_x_dif = self.x_pre[10, 0]
            self.stat_transform_rotation_y_dif = self.x_pre[11, 0]
            self.stat_transform_rotation_z_dif = self.x_pre[12, 0]
            self.stat_transform_rotation_w_dif = self.x_pre[13, 0]
            return 0
        else:
            self.xt = np.array([[self.stat_transform_translation_x], [self.stat_transform_translation_y], [self.stat_transform_translation_z],
                                [self.stat_transform_rotation_x], [self.stat_transform_rotation_y], [self.stat_transform_rotation_z],
                                [self.stat_transform_rotation_w],  [self.stat_transform_translation_x_dif], [self.stat_transform_translation_y_dif],
                                [self.stat_transform_translation_z_dif], [self.stat_transform_rotation_x_dif], [self.stat_transform_rotation_y_dif],
                                [self.stat_transform_rotation_z_dif], [self.stat_transform_rotation_w_dif]])

            self.zt = np.array([[self.transform_translation_x, self.transform_translation_y,
                                 self.transform_translation_z, self.transform_rotation_x, self.transform_rotation_y,
                                 self.transform_rotation_z, self.transform_rotation_w]]).transpose()
            self.Predict()
            self.Update()
            # 结果的保存
            self.stat_transform_translation_x = self.xt[0, 0]
            self.stat_transform_translation_y = self.xt[1, 0]
            self.stat_transform_translation_z = self.xt[2, 0]
            self.stat_transform_rotation_x = self.xt[3, 0]
            self.stat_transform_rotation_y = self.xt[4, 0]
            self.stat_transform_rotation_z = self.xt[5, 0]
            self.stat_transform_rotation_w = self.xt[6, 0]
            self.stat_transform_translation_x_dif = self.xt[7, 0]
            self.stat_transform_translation_y_dif = self.xt[8, 0]
            self.stat_transform_translation_z_dif = self.xt[9, 0]
            self.stat_transform_rotation_x_dif = self.xt[10, 0]
            self.stat_transform_rotation_y_dif = self.xt[11, 0]
            self.stat_transform_rotation_z_dif = self.xt[12, 0]
            self.stat_transform_rotation_w_dif = self.xt[13, 0]

    def Predict(self):
        self.x_pre = np.dot(self.A, self.xt)
        self.P_pre = np.dot(self.A, np.dot(self.Pt, self.A.transpose())) + self.omegat

    def Update(self):
        Mat1 = np.dot(self.H, np.dot(self.P_pre, self.H.transpose())) + self.vt
        Kk = np.dot(self.P_pre, np.dot(self.H.transpose(), np.linalg.inv(Mat1)))
        self.xt = self.x_pre + np.dot(Kk, (self.zt - np.dot(self.H, self.x_pre)))
        Mat2 = np.eye(14) - np.dot(Kk, self.H)
        self.Pt = np.dot(Mat2, np.dot(self.P_pre, Mat2.transpose())) \
                  + np.dot(Kk, np.dot(self.vt, Kk.transpose()))
    
    def get_pose(self):
        pose = [self.stat_transform_translation_x, self.stat_transform_translation_y, self.stat_transform_translation_z,
                self.stat_transform_rotation_x, self.stat_transform_rotation_y, self.stat_transform_rotation_z,
                self.stat_transform_rotation_w]
        return pose_to_SE3(pose)  


