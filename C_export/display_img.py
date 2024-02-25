
import os
from os import listdir
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps 

image_list=[]
label_list=[]
folder_dir = "bmp_traitee"
for images in os.listdir(folder_dir):
    # check if the image ends with png
    if (images.endswith(".bmp")):
        temp=Image.open(folder_dir+"/"+images)
        temp=temp.rotate(180)
        temp=ImageOps.grayscale(temp) 
        image_list.append(temp)
        label_list.append(images.split('_')[0])
        #temp.show()

plt.figure(figsize=(5,5))
for i in range(len(image_list)):
    plt.subplot(10,21,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_list[i], cmap=plt.cm.binary)
    plt.xlabel(label_list[i])
plt.show()