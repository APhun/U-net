from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

'''
x_train = np.ndarray((85,512,512,1), dtype=np.uint8)
y_train = np.ndarray((85,512,512,1), dtype=np.uint8)

x_test = np.ndarray((80,512,512,1), dtype=np.uint8)
y_test = np.ndarray((80,512,512,1), dtype=np.uint8)
'''
x_train = np.ndarray((85,128,128,1), dtype=np.uint8)
y_train = np.ndarray((85,128,128,1), dtype=np.uint8)

x_test = np.ndarray((80,128,128,1), dtype=np.uint8)
y_test = np.ndarray((80,128,128,1), dtype=np.uint8)

count = 0
print('-'*30)
print("loading training data...")
dir = "D:\dataset\data-train\img"
for root, dirs, files in os.walk(dir):
    for file in files:
        name = os.path.join(root,file).split("D:\\dataset\\data-train\\img\\")[1].split(".")[0]
        img = load_img(dir+"\\"+name+".bmp",grayscale = True,target_size=(128,128))
        label = load_img("D:\dataset\data-train\label\\"+name+"_anno.bmp",grayscale = True,target_size=(128,128))
        img = img_to_array(img)
        label = img_to_array(label)
        x_train[count] = img
        y_train[count] = label
        count += 1
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)
print('complete')
print('-'*30)

count = 0
print("loading test data...")
dir = "D:\dataset\data-test\img"
for root, dirs, files in os.walk(dir):
    for file in files:
        name = os.path.join(root,file).split("D:\\dataset\\data-test\\img\\")[1].split(".")[0]
        img = load_img(dir+"\\"+name+".bmp",grayscale = True,target_size=(128,128))
        label = load_img("D:\dataset\data-test\label\\"+name+"_anno.bmp",grayscale = True,target_size=(128,128))
        img = img_to_array(img)
        label = img_to_array(label)
        x_test[count] = img
        y_test[count] = label
        count += 1
np.save('x_test.npy',x_test)
np.save('y_test.npy',y_test)
print('complete')
print('-'*30)
