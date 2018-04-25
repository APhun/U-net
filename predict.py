from keras.model import *

model = load_model('demo.h5')
label_test = model.predict(x_test, batch_size=1, verbose=1)
np.save('y_test_predict.npy', imgs_mask_test)


imgs = np.load('y_test_predict.npy')
imgs *= 255
for i in range(imgs.shape[0]):
    img = imgs[i]
    img = array_to_img(img)
    img.save("../results/%d.jpg"%(i))

