#from data import *
from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.models import *
from keras.optimizers import *
import os


x_train = np.load("x_train.npy")

y_train = np.load("y_train.npy")

x_test = np.load("x_test.npy")

y_test = np.load("y_test.npy")


inputs = Input((128,128,1))

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
print ("conv1 shape:",conv1.shape)

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print ("conv1 shape:",conv1.shape)


pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#print ("pool1 shape:",pool1.shape)
#print ("")

conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#print ("conv2 shape:",conv2.shape)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#print ("conv2 shape:",conv2.shape)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#print ("pool2 shape:",pool2.shape)
#print ("")

conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#print ("conv3 shape:",conv3.shape)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
'''
#print ("conv3 shape:",conv3.shape)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#print ("pool3 shape:",pool3.shape)
#print ("")

conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#print ("conv4 shape:",conv4.shape)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#print ("conv4 shape:",conv4.shape)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#print ("pool4 shape:",pool4.shape)
#print ("")

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#print ("conv5 shape:",conv5.shape)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#print ("conv5 shape:",conv5.shape)
drop5 = Dropout(0.5)(conv5)
#print ("")

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#print ("up6 shape:",up6.shape)
merge6 = add([drop4,up6])
#print ("merge6 shape:",merge6.shape)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#print ("conv6 shape:",conv6.shape)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#print ("conv6 shape:",conv6.shape)
#print ("")

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#print ("up76 shape:",up7.shape)

merge7 = add([conv3,up7])
#print ("merge7 shape:",merge7.shape)

conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#print ("conv7 shape:",conv7.shape)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#print ("conv7 shape:",conv7.shape)
#print ("")
'''

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
#print ("up8 shape:",up8.shape)
merge8 = add([conv2,up8])
#print ("merge8 shape:",merge8.shape)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#print ("conv8 shape:",conv8.shape)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#print ("conv8 shape:",conv8.shape)
#print ("")

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#print ("up9 shape:",up9.shape)
merge9 = add([conv1,up9])
#print ("merge9 shape:",merge9.shape)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#print ("conv9 shape:",conv9.shape)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#print ("conv9 shape:",conv9.shape)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#print ("conv9 shape:",conv9.shape)
#print ("")

conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
#print ("conv10 shape:",conv10.shape)

model = Model(inputs=inputs, outputs = conv10)


model.compile(optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


model.fit(x_train,y_train,epochs=5,batch_size=16)
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)