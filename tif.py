from libtiff import TIFF3D,TIFF
imgdir = TIFF3D.open("train-volume.tif")
imgarr = imgdir.read_image()
for i in range(imgarr.shape[0]):
    imgname = str(i) + ".tif"
    img = TIFF.open(imgname,'w')
    img.write_image(imgarr[i])