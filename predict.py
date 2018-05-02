from keras.models import *
from keras.preprocessing.image import array_to_img
import numpy as np

x_test = np.load('x_test.npy')
model = load_model('demo.h5')  
label_test = model.predict(x_test, batch_size=3, verbose=1)
np.save('yy_test_predict.npy', label_test)


imgs = np.load('y_test_predict.npy')
'''
y_test_predict = y_test_predict.astype('float32')
y_test_predict /= 255
'''

for i in range(imgs.shape[0]):
    
    img = imgs[i]
    img = array_to_img(img)
    img.save("D:\\U-net\\results\\"+str(i)+".tif")

    #print(y_test_predict[i])

'''
for i in range(imgs.shape[0]):
    
    img = imgs[i]
    img = array_to_img(img)
    img.save("D:\\U-net\\results\\"+str(i)+".bmp")
    
    print(imgs[i])
'''