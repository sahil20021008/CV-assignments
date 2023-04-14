# %% [markdown]
# # Section 2

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import open3d
import cv2
import random
import time
import glob

# %%
list_of_paths = []
square_size = 108
numcorners=(8,6)
chessboard_camera_points=np.zeros((36,48,2),np.float64)
i=0
for imgpath in glob.glob("CV-A2-calibration/camera_images/*.jpeg"):
    list_of_paths.append(imgpath)
    img=cv2.imread(imgpath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, numcorners, None)
    if ret == True:
        corners=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001))
        chessboard_camera_points[i]=corners.reshape(48,2)
        cv2.drawChessboardCorners(img, numcorners, corners, ret)
    cv2.namedWindow('Chessboard '+imgpath, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Chessboard '+imgpath, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Chessboard '+imgpath, img)
    cv2.waitKey(1000)
    time.sleep(1)
    i+=1

time.sleep(3)
cv2.destroyAllWindows()

# %%
dist = np.loadtxt("CV-A2-calibration/camera_parameters/distortion.txt")
K = np.loadtxt("CV-A2-calibration/camera_parameters/camera_intrinsic.txt")

# %% [markdown]
# # 2.1

# %%
def func(a,b,c,d):
    return cv2.solvePnPRansac(a,b,c,d,flags=cv2.SOLVEPNP_DLS)
def func2(a):
    return cv2.Rodrigues(a)
offsets=np.zeros((36,1),np.float64)
norme=np.zeros((3,36),np.float64)
lidar_coords=[]
i=0
for imgpath in os.listdir("CV-A2-calibration/lidar_scans"):
    lidar_coords.append(np.asarray(open3d.io.read_point_cloud("CV-A2-calibration/lidar_scans/"+imgpath).points))
    points=lidar_coords[i]
    centroid=np.mean(points,axis=0)
    points=points-centroid
    U,S,V=np.linalg.svd(points)
    normal=V[2,:]
    norme[:,i]=normal
    offsets[i]=-np.dot(normal,centroid)
    i+=1
lidar_coords=np.array(lidar_coords)
norme=norme.T
print("Offsets")
for z in offsets:
    print(z)
print("Normals")
for z in norme:
    print(z)

# %% [markdown]
# # 2.3

# %%
trans_mat_list=np.zeros((36,3,4),np.float64)
i=0
while i<36:
    if lidar_coords[i].shape[0]>48:
        rand=random.randint(0,lidar_coords[i].shape[0]-48)
        _,rvecs,tvecs,_=func(lidar_coords[i][rand:rand+48],chessboard_camera_points[i],K,dist)
    else:
        _,rvecs,tvecs,_=func(lidar_coords[i],chessboard_camera_points[i][:int(lidar_coords[i].shape[0])],K,dist)
    T=np.zeros((3,4),np.float64)
    T[:3,:3]=func2(rvecs)[0]
    T[:3,3]=tvecs.reshape(3)
    trans_mat_list[i]=T
    i+=1
print("Transformation Matrix")
print(np.mean(trans_mat_list,axis=0))

# %% [markdown]
# # 2.4
# 

# %%
i=0
while i<36:
    img=cv2.imread(list_of_paths[i])
    point,_=cv2.projectPoints(lidar_coords[i],func2(trans_mat_list[i][:3,:3])[0],trans_mat_list[i][:3,3],K,dist)
    for j in range(point.shape[0]):
        try:
            cv2.circle(img,(int(point[j,0,0]),int(point[j,0,1])),1,(255,0,0),-1)
        except:
            continue
    cv2.namedWindow('Chessboard '+list_of_paths[i], cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Chessboard '+list_of_paths[i], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Chessboard '+list_of_paths[i], img)
    cv2.waitKey(1000)
    time.sleep(2)
    i+=1
time.sleep(3)
cv2.destroyAllWindows()

# %% [markdown]
# # 2.5

# %%
temp_count=0
num_images=0
for imgpath in os.listdir("CV-A2-calibration/camera_parameters"):
    if temp_count<2:
        temp_count+=1
        continue
    if num_images>=5:
        break
    nci = np.loadtxt("CV-A2-calibration/camera_parameters/" +imgpath+"/camera_normals.txt")
    nli=norme[num_images]
    rnli=np.matmul(trans_mat_list[num_images][:3,:3],nli)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.quiver(0, 0, 0, nci, nli, rnli)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    num_images += 1

# %%
cosine_error_a=np.zeros((36,1),np.float64)
cosine_error_b=np.zeros((36,1),np.float64)
temp_count=0
num_images=0
for imgpath in os.listdir("CV-A2-calibration/camera_parameters/"):
    if temp_count<2:
        temp_count+=1
        continue
    nci = np.loadtxt("CV-A2-calibration/camera_parameters/" +imgpath+"/camera_normals.txt")
    nli=norme[num_images]
    rnli=np.matmul(trans_mat_list[num_images][:3,:3],nli)
    cosine_error_a[num_images]=1-np.dot(nci,nli)/(np.linalg.norm(nci)*np.linalg.norm(nli))
    cosine_error_b[num_images]=1-np.dot(nci,rnli)/(np.linalg.norm(nci)*np.linalg.norm(rnli))
    print("Cosine distance between camera normal and lidar normal: ", cosine_error_a[num_images])
    print("Cosine distance between camera normal and rotated lidar normal: ", cosine_error_b[num_images])
    num_images += 1
print("Mean cosine distance between camera normal and lidar normal: ",np.mean(cosine_error_a))
print("Mean cosine distance between camera normal and rotated lidar normal: ",np.mean(cosine_error_b))
print("Standard deviation cosine distance between camera normal and lidar normal: ",np.std(cosine_error_a))
print("Standard deviation cosine distance between camera normal and rotated lidar normal: ",np.std(cosine_error_b))
plt.hist(cosine_error_a, bins=30)
plt.title("Cosine distance between camera normal and lidar normal")
plt.show()
plt.hist(cosine_error_b, bins=30)
plt.title("Cosine distance between camera normal and rotated lidar normal")
plt.show()


