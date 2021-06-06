import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import cv2
import glob
import numpy as np
from keras.models import *
from keras.layers import *
import image_utils
import scipy.io as sio
from scipy.ndimage import rotate

def predicted_output(model,X_dev, height=68,width=68,n_ops=2):
    	batch_len=X_dev.shape[0]
    	pr = model.predict(X_dev)
    	print(np.array(pr).shape)
	#print(np.array(pr).shape)
    	pr = np.argmax(pr,axis=-1)
    	pred = np.zeros((pr.shape[0],pr.shape[1],height,width),dtype = int)
    	for i in range(n_ops-1):
    	    	pred[:,:,:,:] = ((pr==i+1)*255).astype(int)
    	#sio.savemat(dir_path+'pr_dev.mat', {'y_pred':pred,'name_dev':name})
    	return pred

subs = ['M5']
height = 68
width = 68
n_ops = 2
segnet_model = load_model('/mnt/d/IISc_Internship/segnet_conv2d_seen_3decoders_binary/model_best.weights')
segnet_model.summary()
save_path=('/mnt/d/IISc_Internship/segnet_conv2d_seen_3decoders_binary/Test_Results/')


for sub in range(len(subs)):
	vid = glob.glob( '/mnt/d/IISc_Internship/Test_vids/'+'/*.avi'  )
	vid.sort()
	for vl in range(len(vid)):
		vidcap = cv2.VideoCapture(vid[vl])
		a=[]
		names_test = []
		success = True
		while success:
			success,image = vidcap.read()
			a.append(image)
		a = np.array(a)
		b = np.zeros((a.shape[0]-1,68,68,3))
		for i in range(a.shape[0]-1):
			names_test.append(subs[sub]+'_'+vid[vl]+str(i).zfill(3))
			b[i,:,:,:] = a[i]
		b=b.astype('float64')
		b=b/255
		if not os.path.exists(save_path):
		    print("creating test directory : "+save_path)
		    os.makedirs(save_path)
		else:
		    print("image directory already exists")
		#b_test_rot=np.array(b_test_rot)
		#names_test_rot = np.array(names_test_rot)
		b=np.array(b)
		names_test = names_test[0:len(names_test)]
		names_test = np.array(names_test)
		print(np.array(b).shape)
		print(np.array(names_test).shape)
		preds = predicted_output(segnet_model,b,height,width,n_ops=2)
		sio.savemat(save_path+subs[sub]+'_'+vid[vl][len(vid[vl])-7:len(vid[vl])-4]+'.mat', {'pred':preds,'name':names_test})







Test_Vid=['413']
v=0

for sub in range(len(subs)):
	vid = glob.glob( '/mnt/d/IISc_Internship/M5_Vids/M5_'+Test_Vid[v]+'.avi'  )
	vid.sort()
	for vl in range(len(vid)):
		vidcap = cv2.VideoCapture(vid[vl])
		a=[]
		names_test = []
		success = True
		while success:
			success,image = vidcap.read()
			a.append(image)
		a = np.array(a)
		b = np.zeros((a.shape[0]-1,68,68,3))
		for i in range(a.shape[0]-1):
			names_test.append(subs[sub]+'_'+vid[vl]+str(i).zfill(3))
			b[i,:,:,:] = a[i]
		b=b.astype('float64')
		b=b/255
		if not os.path.exists(save_path):
		    print("creating test directory : "+save_path)
		    os.makedirs(save_path)
		else:
		    print("image directory already exists")
		#b_test_rot=np.array(b_test_rot)
		#names_test_rot = np.array(names_test_rot)
		b=np.array(b)
		names_test = names_test[0:len(names_test)]
		names_test = np.array(names_test)
		print(np.array(b).shape)
		print(np.array(names_test).shape)
		preds = predicted_output(segnet_model,b,height,width,n_ops=2)
		sio.savemat(save_path+subs[sub]+'_'+Test_Vid[v]+'.mat', {'pred':preds,'name':names_test})
