# %% [markdown]
# # Section 1
# # 1.1

# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob 
import os
import seaborn as sns
import pandas as pd
import random
import time

# %%
obj_points=np.zeros((35,3),np.float32)
obj_points[:,:2]=np.mgrid[0:7,0:5].T.reshape(-1,2)
world_points=[]
image_points=[]
for imgpath in glob.glob("cv_data_1/*.jpg"):
    img=cv2.imread(imgpath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(7,5),None)
    if ret==True:
        world_points.append(obj_points)
        corners=cv2.cornerSubPix(gray,corners,(9,9),(-1,-1),criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.001))
        image_points.append(corners)
        cv2.drawChessboardCorners(img,(7,5),corners,ret)
        cv2.namedWindow("img",cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("img",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("img",img)
        cv2.waitKey(500)
        time.sleep(1)
time.sleep(1)
cv2.destroyAllWindows()
ret,K,dist,rvecs,tvecs=cv2.calibrateCamera(world_points,image_points,gray.shape[::-1],None,None)
print("Intrinsic Matrix: ")
print(K)
print("Skew: ", K[0][1])
print("Focal Length: ", K[0][0], K[1][1])
print("Principal Point: ", K[0][2], K[1][2])
for i in range(len(rvecs)):
    print("For Image ", i+1, ": ")
    print("Rotation Vector: ", rvecs[i],sep="\n")
    print("Translation Vector: ", tvecs[i],sep="\n")
print("Radial Distortion Coefficients: ")
print(dist)

# %% [markdown]
# # 1.3

# %%
i=1
temp_count=0
for imgpath in glob.glob("cv_data_1/*.jpg"):
    if temp_count>=5:
        break
    temp_count+=1
    img=cv2.imread(imgpath)
    newK,_=cv2.getOptimalNewCameraMatrix(K,dist,(img.shape[1],img.shape[0]),1,(img.shape[1],img.shape[0]))
    undist=cv2.undistort(img,K,dist,None,newK)
    cv2.imwrite("undistorted"+str(i)+".jpg",undist)
    reimg=cv2.resize(undist,(int(undist.shape[1]*0.6),int(undist.shape[0]*0.6)))
    cv2.namedWindow("undistorted",cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("undistorted",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("undistorted",reimg)
    cv2.waitKey(500)
    time.sleep(1)
    i+=1
time.sleep(1)
cv2.destroyAllWindows()

# %% [markdown]
# # 1.4

# %%
reprojection_error=np.zeros((len(image_points)),np.float64)
for i in range(len(image_points)):
    points,_=cv2.projectPoints(world_points[i],rvecs[i],tvecs[i],K,dist)
    error=cv2.norm(image_points[i],points,cv2.NORM_L2)/len(points)
    reprojection_error[i]=error
print("Reprojection Error: ")
print(reprojection_error)
print("Average Reprojection Error: ", reprojection_error.mean())
print("Standard Deviation of Reprojection Error: ", reprojection_error.std())
plt.figure(figsize=(10,5))
sns.set_style("darkgrid")
sns.barplot(x=np.arange(1,26),y=reprojection_error)
plt.ylabel("Reprojection Error")
plt.xlabel("Image Number")
plt.title("Reprojection Error per Image")
plt.show()

# %% [markdown]
# # 1.5

# %%
i=0
for imgpath in glob.glob("cv_data_1/*.jpg"):
    img1=cv2.imread(imgpath)
    img2=cv2.imread(imgpath)
    cv2.drawChessboardCorners(img1,(7,5),image_points[i],True)
    points,_=cv2.projectPoints(world_points[i],rvecs[i],tvecs[i],K,dist)
    cv2.drawChessboardCorners(img2,(7,5),points,True)
    reimg1=cv2.resize(img1,(int(img1.shape[1]*0.6),int(img1.shape[0]*0.6)),interpolation=cv2.INTER_AREA)
    reimg2=cv2.resize(img2,(int(img2.shape[1]*0.6),int(img2.shape[0]*0.6)),interpolation=cv2.INTER_AREA)
    horizontal=np.concatenate((reimg1,reimg2),axis=1)
    cv2.namedWindow("img",cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("img",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("img",horizontal)
    cv2.waitKey(500)
    time.sleep(1)
    i+=1
time.sleep(1)
cv2.destroyAllWindows()

# %% [markdown]
# # 1.6

# %%
world=np.zeros((35,3),np.float32)
norme=np.zeros((3,25),np.float64)
world[:,:2]=np.mgrid[0:7,0:5].T.reshape(-1,2)*0.03
i=0
while i<25:
    _,rvecs,tvecs,_=cv2.solvePnPRansac(world,image_points[i],K,None)
    w2=np.array([0,0.03,0])
    w1=np.array([0.03,0,0])
    wn=np.cross(w1,w2)/np.linalg.norm(np.cross(w1,w2))
    camn=np.dot(cv2.Rodrigues(rvecs)[0],wn)/np.linalg.norm(np.dot(cv2.Rodrigues(rvecs)[0],wn))
    norme[:,i]=camn
    i+=1
norme=norme.T
print("Normal Vectors:")
for i in norme:
    print(i)


