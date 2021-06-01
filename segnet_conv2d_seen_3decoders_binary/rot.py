import cv2
from scipy.ndimage import rotate
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

path = '/home/rgukt/Desktop/IISc/annotation/files_for_navaneetha_correction/videos/F1/F1_342.avi'
vidcap = cv2.VideoCapture(path)
success = True
angles = [0,90,180,270]
angles1 = [0,-270,180,-90]
a =[]
c=0
mat =loadmat('/home/rgukt/Desktop/IISc/navaneetha_segnet/Important/data/mat_110/F1/342.mat')
masks = ['mask1','mask2','mask3']
for i in range(len(mat['masks']['mask1'][0])):
	success, image = vidcap.read()
	a.append(image)
	#cv2.imshow('check',image)
	#cv2.waitKey(10)
seg_labels = np.zeros((68, 68, 3), dtype = 'uint8')
for j in range(3):
	#print(sum(sum(mat['masks'][masks[j]][0][0].astype('uint8'))))
	seg_labels[:,:,j] = mat['masks'][masks[j]][0][0].astype('uint8')
#cv2.imshow('c',seg_labels)
#cv2.waitKey(1000)
for i in range(len(angles)):
	#print(c)
	#print(i)
	rot_in = rotate(a[0],0, reshape=False)
	rot_op = rotate(seg_labels*255,-angles[i], reshape=False)
	rot_op1 = rotate(rot_op,angles1[i], reshape=False)
	fig, ax = plt.subplots(1, 3)
	ax[0].imshow(rot_in)
	ax[1].imshow(rot_op)
	ax[2].imshow(rot_op1)
	plt.show()
c=c+1

