import numpy as np
import copy 
import cv2
import matplotlib.pyplot as plt
import math

# 行列数据为均值为0剔除处理
# data为获得的深度数据
def col_row_process(data):
    data_new = data
    #列数据处理
    col = 0
    d = data.shape[1]
    while col < d:
        da = data_new[:, col]  # 获取某列的值,data[0,:]获取某行的值
        if np.mean(da) == 0:
            data_new = np.delete(data_new, col, 1)
            d -= 1
        else:
            col += 1
    #行数据处理
    row = 0
    d = data_new.shape[0]
    while row < d:
        da = data_new[row, :]  # 获取某行的值
        if np.mean(da) == 0:
            data_new = np.delete(data_new, row, 0)
            d -= 1
        else:
            row += 1
    return data_new



def interp_data(depth_image):
    fn = 0
    new_data = depth_image
    try:
        for i in range(new_data.shape[0]):
            for j in range(new_data.shape[1]):
                if new_data[i,j]==0:
                    if j==0:
                        d = j
                        while new_data[i,d]==0:
                            d +=1
                        new_data[i, j] = new_data[i, d]
                    else:
                        new_data[i, j] = new_data[i, j - 1]
                    fn += 1
        return new_data,fn             
    except Exception as e:
        print(e)


def abnormal_process(depth_image):
    depth_data = depth_image.copy()
    replacements_made = 0

    # Row-wise processing
    for i in range(len(depth_data)):
        row_data = depth_data[i]
        row_mean = np.mean(row_data)
        row_std = np.std(row_data)
        up_data = row_mean + (2.5 * row_std)
        lo_data = row_mean - (1.5 * row_std)

        for j in range(len(row_data)):
            if row_data[j] > up_data or row_data[j] < lo_data:
                if j == 0:
                    # Search forward for a valid data point
                    d = j
                    while d < len(row_data) and row_data[d] > up_data:
                        d += 1
                    if d < len(row_data):
                        row_data[j] = row_data[d]
                    else:
                        row_data[j] = row_data[j-1]  # Fall back to previous value if no valid data
                else:
                    row_data[j] = row_data[j-1]  # Use the previous value if it's out of range
                replacements_made += 1

    # Column-wise processing
    for j in range(depth_data.shape[1]):
        col_data = np.sort(depth_data[:, j])
        col_mean = np.mean(col_data)
        col_std = np.std(col_data)
        up_data = col_mean + (2.5 * col_std)
        lo_data = col_mean - (1.5 * col_std)

        for i in range(len(depth_data[:, j])):
            if depth_data[i, j] > up_data or depth_data[i, j] < lo_data:
                if i == 0:
                    # Search forward for a valid data point
                    d = i
                    while d < len(depth_data[:, j]) and depth_data[d, j] > up_data:
                        d += 1
                    if d < len(depth_data[:, j]):
                        depth_data[i, j] = depth_data[d, j]
                    else:
                        depth_data[i, j] = depth_data[i-1, j]  # Fall back to previous value if no valid data
                else:
                    depth_data[i, j] = depth_data[i-1, j]  # Use the previous value if it's out of range
                replacements_made += 1
    print(f"Replacements made: {replacements_made}")

    return depth_data, replacements_made



def project_to_3d(points, depth, intrinsics, show=True,resize = True,sequence = "yx"):
    '''
    args:
        points: list of tuples of pixel coordinates
        depth: depth image
        intrinsics: list of camera intrinsics
        show: bool, whether to show the image with the points plotted
        resize: bool, whether to resize the depth image
        sequence: string, the sequence of the pixel coordinates, either "yx" or "xy"
    return:
        points_3d: list of tuples of 3D coordinates
    
    '''
    if resize:
        depth = cv2.resize(depth, (320,240)) #??
    if show:
        plt.imshow(depth)
    
    points_3d = list()
    for x,y in points:
        if sequence == "yx":
            x,y = y,x
        elif sequence == "xy":
            x,y = x,y

        x = math.floor(x) 
        y = math.floor(y)
        d = depth[y][x]        
        # Plot points (x, y) on the image
        if show:
            if d>0:
                plt.scatter(x, y, color='blue', s=10)  # Adjust the size (s) as needed
            else:
                plt.scatter(x, y, color='red', s=10)
        # z = d / depth_scale
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # 3d point in meter
        z = d / 1000
        fx, fy, cx, cy = intrinsics  # Assuming intrinsics is [fx, fy, cx, cy]
        
        # Calculate 3D coordinates
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        
        if show:
            print(f'x:{x} \t y:{y} \t z:{z}')
        points_3d.append((x,y,z)) #in ee frame in this case ee frame is the same

        # if x+y+z ==0:
        #     print("x y z are 0")
        #     print("z is",z)
        #     print("depth is",depth)
        #     input("check project to 3d class")
        
    if show:
        plt.axis('off')  # Turn off axis labels
        plt.show()
    
    return points_3d
