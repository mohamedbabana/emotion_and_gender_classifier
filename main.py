"""
This programme is used to detect, in a real time, the gender and the emotion of peaple in front of one camera connected to the camputer
"""

from fun import *
import matplotlib.pyplot as plt

####### set of the indices of all camera connected to the camputer
cam_indices = ConnectCam()

####### for don't use the default webcam
#cam_indices.pop(0) 

####### number of camera used
n = len(cam_indices)

####### Set of all cameras used
cams = []
for i in range(n):
    cams.append(cv2.VideoCapture(cam_indices[i]))

w, h = 400, 300
for i in range(n):
    ##### Configure the height and the wight of the frame
    cams[i].set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cams[i].set(cv2.CAP_PROP_FRAME_HEIGHT, h)

####### Check if there are cameras connected
if n == 0:
    print("Aucune caméra branchée !")
else:

    while True :
        ######## Set of one capture from all cameras
        frames = []
        for i in range(n):
            frames.append(cams[i].read()[1])
            ####### applid the fonction cam to eatch frame to find faces, emotion and genders of peaple in each frame 
            cam(frames[i], h, w)
            ####### Show eatch frame after its treatment
            cv2.imshow('frame'+ str(i+1), frames[i])

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for i in range(n):
        ####### take off all camera
        cams[i].release()
    ######## destroy all frames
    cv2.destroyAllWindows()
