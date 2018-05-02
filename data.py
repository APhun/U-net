from keras.preprocessing.image import img_to_array, load_img, array_to_img,ImageDataGenerator
import numpy as np
import os


train_num = 30
test_num = 30

x_train = np.ndarray((train_num,512,512,3), dtype=np.uint8)
y_train = np.ndarray((train_num,512,512,1), dtype=np.uint8)

x_test = np.ndarray((test_num,512,512,1), dtype=np.uint8)
y_test = np.ndarray((test_num,512,512,1), dtype=np.uint8)



count = 0
print('-'*30)
print("loading training data...")
dir = "D:\dataset\data-train\img"
for root, dirs, files in os.walk(dir):
    for file in files:
        name = os.path.join(root,file).split("D:\\dataset\\data-train\\img\\")[1].split(".")[0]
        img = load_img(dir+"\\"+name+".tif",target_size=(512,512))
        label = load_img("D:\dataset\data-train\label\\"+name+".tif",grayscale = True,target_size=(512,512))
        img = img_to_array(img)
        label = img_to_array(label)
        x_train[count] = img
        y_train[count] = label
        #print(x_train)
        count += 1
print('complete')
print('-'*30)


print('start augmenting...')
 #将标签作为训练集的一个通道
x_train[:,:,:,2] = y_train[:,:,:,0]
# print(x_train[:,:,:,2])
#np.save('x_train.npy',x_train)
#np.save('y_train.npy',y_train)

for count in range(train_num):
    img = array_to_img(x_train[count])
    img.save("D:\dataset\data-augmentation\\raw\\"+str(count)+".tif")


img_generator = ImageDataGenerator(rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

#x_train = np.load("x_train.npy")
#print(x_train[:,:,2])

i = 0 # 数据增强
for num in range(train_num):
    for batch in img_generator.flow(x_train,batch_size=1,
                       save_to_dir="D:\dataset\data-augmentation\\unsplited",
                       save_prefix="train",
                       save_format='tif'
                       ):
        i += 1
        if i >= 50:
            i = 0
            break


dir ="D:\dataset\data-augmentation\\unsplited"
img_num = 0
for num in os.listdir(dir): #fn 表示的是文件名
        img_num += 1


new_img = np.ndarray((img_num,512,512,1), dtype=np.uint8)
new_label = np.ndarray((img_num,512,512,1), dtype=np.uint8)


count = 0
for root, dirs, files in os.walk(dir):
    for file in files:
        # name = os.path.join(root,file).split("D:\\dataset\\data-train\\img\\")[1].split(".")[0]
        # img = load_img(dir+"\\"+name+".bmp",target_size=(128,128))
        # label = load_img("D:\dataset\data-train\label\\"+name+"_anno.bmp",grayscale = True,target_size=(128,128))
        img = load_img(os.path.join(root,file))
        img = img_to_array(img)
        label = img[:,:,2]
        
        label = label.reshape((512,512,1))
        
        new_label[count] = label
        label[label != 0] = 255
        label = array_to_img(label)
        
        label.save("D:\dataset\data-augmentation\label\\"+str(count)+".tif")
        
        img[:,:,2] = 0
                
        img = array_to_img(img)
        img.save("D:\dataset\data-augmentation\img\\"+str(count)+".tif")
        img = load_img("D:\dataset\data-augmentation\img\\"+str(count)+".tif",grayscale=True)
        img = img_to_array(img)
        
        new_img[count] = img
        #print(new_img[count].shape)
        count += 1

#np.save('x_train.npy',new_img)
#np.save('y_train.npy',new_label)

np.save('x_train.npy',new_img)

np.save('y_train.npy',new_label)

count = 0
print("loading test data...")
dir = "D:\dataset\data-test\img"
for root, dirs, files in os.walk(dir):
    for file in files:
        name = os.path.join(root,file).split("D:\\dataset\\data-test\\img\\")[1].split(".")[0]
        img = load_img(dir+"\\"+name+".tif",grayscale = True,target_size=(512,512))
        #label = load_img("D:\dataset\data-test\label\\"+name+"_anno.bmp",grayscale = True,target_size=(512,512))
        img = img_to_array(img)
        #label = img_to_array(label)
        #label[label != 0] = 255
        x_test[count] = img
        #y_test[count] = label
        count += 1
        
np.save('x_test.npy',x_test)

#np.save('y_test.npy',y_test)

print('complete')
print('-'*30)
