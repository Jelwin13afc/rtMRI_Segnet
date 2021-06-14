import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto
#config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import glob
#import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import imread
from tensorflow.keras import optimizers
import model
import image_utils
import scipy.io as sio


##IMAGE_ORDERING = 'channels_last'
#root_dir = '/home/sgeadmin/renuka/data/'
#segs_path=[root_dir+'ground_truth/contour1/',root_dir+'ground_truth/contour2/',root_dir+'ground_truth/contour3/']
#images_path=root_dir+'input/'
#batch_size=[len(os.listdir(images_path))]

subs = ['F1','F2','F3','F4','F5','M1','M2','M3','M4','M5']

#model_dir = '/home/user/home2/models/navaneetha/segnet_conv2d_seen_3decoders_binary/'
model_dir = '/mnt/d/IISc_Internship/segnet_conv2d_seen_3decoders_binary/'
#in_dir = '/home/user/home2/data/navaneetha/110_videos/' ## videos directory
#out_dir = '/home/user/home2/data/navaneetha/mat_110/' ## masks matfiles directory
in_dir = '/mnt/d/IISc_Internship/Videos/' ## videos directory
out_dir = '/mnt/d/IISc_Internship/Masks/' ## masks matfiles directory
##mat_dir = '/home2/data/navaneetha/fixed_c1.mat'
n_ops = 2
height=68
width=68
epoch =30
new = 1
#segnet_model=model.Segnet(n_ops ,height, width)
#for layer in segnet_model.layers:
    #print(layer.output_shape)
if not os.path.exists(model_dir):
    print("creating model directory ")
    os.makedirs(model_dir)
actual_set=np.array([342,391,392,393,394,395,397,398,399,406,413])
train_matrix = actual_set[0:9]
val_matrix = actual_set[9:10]
test_matrix= actual_set[10:11]
X_train,Y_train,X_val,Y_val,X_test,Y_test,name,name_val, names_test=image_utils.imageSegmentationGenerator(in_dir,out_dir,n_ops,height,width,train_matrix,val_matrix,test_matrix)
print ('X_train.shape',X_train.shape)
print ('Y_train[0].shape',Y_train[0].shape)
print (Y_train[1].shape)
print (Y_train[2].shape)
print (X_val.shape)
print (Y_val[0].shape)
print (Y_val[1].shape)
print (Y_val[2].shape)
print (X_test.shape)
print (Y_test[0].shape)
print (Y_test[1].shape)
print (Y_test[2].shape)
print (name.shape)
print (name_val.shape)

print('*********new model fitting*********')
segnet_model=model.Segnet(n_ops ,height, width)
segnet_model.summary()
segnet_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('compilation completed')
callbacks = [ModelCheckpoint(model_dir+'model_best.weights', monitor='val_loss', verbose=0, save_best_only=True, mode = 'auto',period=1)]
print('callbacks completed')
history=segnet_model.fit(X_train, Y_train, epochs=epoch,batch_size=2,validation_data = (X_val, Y_val),verbose=1,callbacks=callbacks)
score = segnet_model.evaluate(X_test, Y_test)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
