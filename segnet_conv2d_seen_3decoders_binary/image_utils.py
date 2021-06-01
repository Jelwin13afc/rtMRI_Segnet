import cv2
import glob
import numpy as np
import os
import sys
import scipy.io as sio
from scipy.ndimage import rotate
import random
def generate_data(in_dir,out_dir,sub,vid,X_train, Y_train,names,flag,height,width,frames,n_ops):
	#temp_mat = sio.loadmat(mat_dir)
	mat = sio.loadmat(out_dir+sub+'/'+str(vid)+'.mat')
	vidcap = cv2.VideoCapture(in_dir+sub+'/'+sub+'_'+str(vid)+'.avi')
	success = True
	x_train_r = []
	y_train = [[],[],[]]
	for i in range(len(mat['masks']['mask1'][0])):
		success,image = vidcap.read()
		x_train_r.append(image)
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
		y_train[0].append(seg_labels1)
		y_train[1].append(seg_labels2)
		y_train[2].append(seg_labels3)
	x_train = np.zeros((np.array(x_train_r).shape[0],68,68,3))
	for ng in range(np.array(x_train_r).shape[0]):
		x_train[ng,:,:,:] = x_train_r[ng]
	x_train = list(x_train)
	l = len(x_train)
	fr = int(frames/2)
	for k in range(fr-(l%fr)): ## to adjust at boundary 24/2 = 12 for 0.5 sec shift
		x_train.append(x_train[-1])
		y_train[0].append(y_train[0][-1])
		y_train[1].append(y_train[1][-1])
		y_train[2].append(y_train[2][-1])
	l = len(x_train)
	if (flag == 1): ## for train data and vadlidation data
		li = list(range(0,l+1,fr)) ## one second duration with 0.5 sec shift
		for m in range(len(li)-2):
			X_train.append(x_train[li[m]:li[m+2]])
			Y_train[0].append(y_train[0][li[m]:li[m+2]])
			Y_train[1].append(y_train[1][li[m]:li[m+2]])
			Y_train[2].append(y_train[2][li[m]:li[m+2]])
			names.append(sub+'_'+str(vid)+'_'+str(li[m]+1).zfill(3)+'_'+str(li[m+2]).zfill(3))
	elif (flag == 0):
		li = list(range(0,l+1,frames)) ## one second duration with no shift
		for m in range(len(li)-1):
			X_train.append(x_train[li[m]:li[m+1]])
			Y_train[0].append(y_train[0][li[m]:li[m+1]])
			Y_train[1].append(y_train[1][li[m]:li[m+1]])
			Y_train[2].append(y_train[2][li[m]:li[m+1]])
			names.append(sub+'_'+str(vid)+'_'+str(li[m]+1).zfill(3)+'_'+str(li[m+1]).zfill(3))
	else:
		print("please mention train or test data")
	return X_train, Y_train, names


def imageSegmentationGenerator( in_dir , out_dir , n_ops , frames, height , width , train_matrix, val_matrix, test_matrix):
	subs = ['M5']
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
	names_test = []
	### To generate train data
	for sub in subs:
		for vid in train_matrix:
			X_train, Y_train, names = generate_data(in_dir,out_dir,sub,vid,X_train, Y_train,names,1,height,width,frames,n_ops)
		for vid in val_matrix:
		        X_val, Y_val, names_val = generate_data(in_dir,out_dir,sub,vid,X_val, Y_val, names_val,1,height,width,frames,n_ops)
		for vid in test_matrix:
		        X_test, Y_test, names_test = generate_data(in_dir,out_dir,sub,vid,X_test,Y_test, names_test,0,height,width,frames,n_ops)
	### To generate test data
	X_train = np.array(X_train).astype('float64')
	X_train = X_train/255
	X_test = np.array(X_test).astype('float64')
	X_test = X_test/255
	X_val = np.array(X_val).astype('float64')
	X_val = X_val/255
	Y_train1 = [np.array(Y_train[i]) for i in range(3)]
	Y_val1 = [np.array(Y_val[i]) for i in range(3)]
	Y_test1 = [np.array(Y_test[i]) for i in range(3)]
	return np.array(X_train), Y_train1, np.array(X_val), Y_val1, np.array(X_test), Y_test1, np.array(names), np.array(names_val), np.array(names_test)

def predicted_output(model,X_dev,frames = 24, height=68,width=68,n_ops=2):
    	batch_len=X_dev.shape[0]
    	pr = model.predict(X_dev)
        #print(np.array(pr).shape)
    	pr = np.argmax(pr,axis=-1)
    	pred = np.zeros((pr.shape[0],frames,height,width),dtype = int)
    	for i in range(n_ops-1):
    	    	pred[:,:,:,:] = ((pr==i+1)*255).astype(int)
    	#sio.savemat(dir_path+'pr_dev.mat', {'y_pred':pred,'name_dev':name})
    	return pred
