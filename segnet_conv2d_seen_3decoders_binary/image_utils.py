import cv2
import glob
import numpy as np
import os
import sys
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from matplotlib import pyplot as plt
import random
def generate_data(in_dir,out_dir,sub,vid,X_train, Y_train,names,height,width,n_ops):
	#temp_mat = sio.loadmat(mat_dir)
	mat = sio.loadmat(out_dir+sub+'/'+str(vid)+'.mat')
	vidcap = cv2.VideoCapture(in_dir+sub+'/'+sub+'_'+str(vid)+'.avi')
	print(vidcap)
	success = True
	for i in range(len(mat['masks']['mask1'][0])):
		success,image = vidcap.read()
		X_train.append(image)
		seg_labels1 = np.zeros((height, width, n_ops), dtype = int)
		seg_labels2 = np.zeros((height, width, n_ops), dtype = int)
		seg_labels3 = np.zeros((height, width, n_ops), dtype = int)
		for j in range(n_ops-1):
			seg_labels1[:,:,j+1] = mat['masks']['mask1'][0][i].astype(int)
			seg_labels2[:,:,j+1] = mat['masks']['mask2'][0][i].astype(int)
			seg_labels3[:,:,j+1] = mat['masks']['mask3'][0][i].astype(int)
		seg_labels1[:,:,0] = ((seg_labels1[:,:,1] == 0)).astype(int)
		seg_labels2[:,:,0] = ((seg_labels2[:,:,1] == 0)).astype(int)
		seg_labels3[:,:,0] = ((seg_labels3[:,:,1] == 0)).astype(int)
		#print(len(y_train))
		#print(np.array(x_train_r).shape)
		#x_train[i,:,:,:] = image
		Y_train[0].append(seg_labels1)
		Y_train[1].append(seg_labels2)
		Y_train[2].append(seg_labels3)
		names.append(sub+'_'+str(vid)+'_'+str(i+1).zfill(3))
	return X_train, Y_train, names


def imageSegmentationGenerator( in_dir , out_dir , n_ops , height , width , train_matrix, val_matrix, test_matrix):
	subs = ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5']
	assert in_dir[-1] == '/'
	assert out_dir[-1][-1] =='/'
	X_train = []
	Y_train = [[],[],[]]
	X_val = []
	Y_val = [[],[],[]]
	X_test = []
	Y_test = [[],[],[]]
	names = []
	names_val = []
	names_test=[]
	### To generate train data
	for sub in subs:
		for vid in train_matrix:
			X_train, Y_train, names = generate_data(in_dir,out_dir,sub,vid,X_train, Y_train,names,height,width,n_ops)
		for vid in val_matrix:
		        X_val, Y_val, names_val = generate_data(in_dir,out_dir,sub,vid,X_val, Y_val, names_val,height,width,n_ops)
		for vid in test_matrix:
		        X_test, Y_test, names_test = generate_data(in_dir,out_dir,sub,vid,X_test, Y_test, names_test,height,width,n_ops)
	#print(len(X_train))
	#print(np.array(X_train).shape)
	X_train = np.array(X_train).astype('float64')
	X_train = X_train/255
	X_val = np.array(X_val).astype('float64')
	X_val = X_val/255
	X_test = np.array(X_test).astype('float64')
	X_test = X_test/255
	Y_train1 = [np.array(Y_train[i]) for i in range(3)]
	Y_val1 = [np.array(Y_val[i]) for i in range(3)]
	Y_test1 = [np.array(Y_test[i]) for i in range(3)]
	return np.array(X_train), Y_train1, np.array(X_val), Y_val1, np.array(X_test), Y_test1, np.array(names), np.array(names_val), np.array(names_test)

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
