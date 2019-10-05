# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:47:39 2019

@author: MUKHESH
"""

import os,signal
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import numpy as np
import random

path='C:/Users/MUKHESH/Documents/Python Scripts/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained=InceptionV3(input_shape=(150,150,3),include_top=False,weights=None)
pre_trained.load_weights(path)
for  layer in pre_trained.layers:
    layer.trainable=False
    
last_layer=pre_trained.get_layer('mixed7')
input_layer=last_layer.output
file='C:/Users/MUKHESH/Downloads/cats_and_dogs_filtered.zip'
if(os.path.isdir('C:/Users/MUKHESH/Documents/Python Scripts/image-classifier/cats_and_dogs_filtered')==0):
    zip_ref=zipfile.ZipFile(file,'r')
    zip_ref.extractall('C:/Users/MUKHESH/Documents/Python Scripts/image-classifier')
    zip_ref.close()

base_dir='C:/Users/MUKHESH/Documents/Python Scripts/image-classifier/cats_and_dogs_filtered'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')
validation_dogs_dir=os.path.join(validation_dir,'dogs')
validation_cats_dir=os.path.join(validation_dir,'cats')

cats_train=os.listdir(train_cats_dir)
dogs_train=os.listdir(train_dogs_dir)
cats_validation=os.listdir(validation_cats_dir)
dogs_validation=os.listdir(validation_dogs_dir)
#print(cats_train[:10])
#print(dogs_train[:10])
#print(cats_validation[:10])
#print(dogs_validation[:10])

n_rows=4
n_cols=4

fig=plt.gcf()
fig.set_size_inches(n_rows*4,n_cols*4)
pic_index=8

next_cats_pic=[os.path.join(train_cats_dir,fname) for fname in cats_train[pic_index-8:pic_index]]
next_dogs_pic=[os.path.join(train_dogs_dir,fname) for fname in dogs_train[pic_index-8:pic_index]]

#for i,img in enumerate(next_dogs_pic+next_cats_pic):
#    sub=plt.subplot(n_rows,n_cols,i+1)
#    sub.axis('Off')
#    im=mpimg.imread(img)
#    plt.imshow(im)
#    
#plt.show()

#inputmap=layers.Input(shape=(150,150,3))
#
#x=layers.Conv2D(16,3,activation='relu')(inputmap)
#x=layers.MaxPooling2D(2)(x)
#x=layers.Conv2D(32,3,activation='relu')(x)
#x=layers.MaxPooling2D(2)(x)
#x=layers.Conv2D(64,3,activation='relu')(x)
#x=layers.MaxPooling2D(2)(x)
x=layers.Flatten()(input_layer)   
#x=layers.Flatten()(x)
x=layers.Dense(1024,activation='relu')(x)
x=layers.Dropout(0.2)(x)#=>this is useful for the regularisation dropout
output=layers.Dense(1,activation='sigmoid')(x)
model=Model(pre_trained.input,output)
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

#train_gen=ImageDataGenerator(rescale=(1./255))
#for avoiding overfitting we use the data augmentation
train_gen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,horizontal_flip=True,fill_mode='nearest')
#img_path=os.path.join(train_cats_dir,cats_train[2])
#img=load_img(img_path,target_size=(150,150))
#x=img_to_array(img)
#x=x.reshape((1,)+x.shape)
#i=0
#for y in train_gen.flow(x,batch_size=1):
#    plt.figure(i)
#    print(y[0])
#    #img=mpimg.imread(array_to_img(y[0]))
#    im=plt.imshow(array_to_img(y[0]))
#    i+=1
#    if i%5==0:
#        break

validation_gen=ImageDataGenerator(rescale=(1./255))
train_generator=train_gen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validation_generator=validation_gen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=2,validation_data=validation_generator,validation_steps=50,verbose=2)
#
#successive_outputs=[layer.output for layer in model.layers[1:]]
#visualization_model=Model(inputmap,successive_outputs)
#
## Let's prepare a random input image of a cat or dog from the training set.
#cat_img_files = [os.path.join(train_cats_dir, f) for f in cats_train]
#dog_img_files = [os.path.join(train_dogs_dir, f) for f in dogs_train]
#img_path = random.choice(cat_img_files + dog_img_files)
#
#img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
#x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
#x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
#
## Rescale by 1/255
#x /= 255
#
#successive_map=visualization_model.predict(x)
#
#layer_names = [layer.name for layer in model.layers]
#for layer_name, feature_map in zip(layer_names, successive_map):
#  if len(feature_map.shape) == 4:
#    # Just do this for the conv / maxpool layers, not the fully-connected layers
#    n_features = feature_map.shape[-1]  # number of features in feature map
#    # The feature map has shape (1, size, size, n_features)
#    size = feature_map.shape[1]
#    # We will tile our images in this matrix
#    display_grid = np.zeros((size, size * n_features))
#    for i in range(n_features):
#      # Postprocess the feature to make it visually palatable
#      x = feature_map[0, :, :, i]
#      x -= x.mean()
#      x /= x.std()
#      x *= 64
#      x += 128
#      x = np.clip(x, 0, 255).astype('uint8')
#      # We'll tile each filter into this big horizontal grid
#      display_grid[:, i * size : (i + 1) * size] = x
#    # Display the grid
#    scale = 20. / n_features
#    plt.figure(figsize=(scale * n_features, scale))
#    plt.title(layer_name)
#    plt.grid(False)
#    plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
#
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epoch=range(len(acc))
#
plt.plot(epoch,acc)
plt.plot(epoch,val_acc)
plt.title('training accuracy vs validation accuracy')
plt.figure()
plt.plot(epoch,loss)
plt.plot(epoch,val_loss)
plt.title('training accuracy vs validation accuracy')
#
model.save('image_model.h5')
os.kill(os.getpid(),signal.SIGKILL)
