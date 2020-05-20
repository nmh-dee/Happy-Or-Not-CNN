#load the required packages
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline

#Normalize the dataset
X_train_orig,Y_train_orig, X_test_orig, Y_test_orig, classes= load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#Create model
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input= Input(input_shape)
    
    X= ZeroPadding2D((3,3))(X_input)
    
    X= Conv2D(32,(7,7),strides=(1,1),name= 'conv0')(X)
    X= BatchNormalization(axis=3,name='bn0')(X)
    X= Activation('relu')(X)
    X= MaxPooling2D((2,2),name='pool1')(X)
    X= Flatten()(X)
    X= Dense(1,activation='sigmoid',name='fc')(X)
    model= Model(input=X_input,outputs=X,name='HappyModel')
    
    return model


happyModel = HappyModel(X_train.shape[1:])   #except the number of example 

#compile the model
happyModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])

#train the model
happyModel.fit(X_train,Y_train,epochs=40,batch_size=64)

#evaluate model use test file
preds = happyModel.evaluate(X_test,Y_test,batch_size=32,verbose=1, sample_weight=None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
###Test with my image
img_path = 'images/luong_smile.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))

#prints the details of my layers in a table with the size of its inputs/outputs
happyModel.summary()

#plots my graph in a nice layout. You can save it as ".png" using SVG(). It is saved in "File" then "Open " it upper
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
