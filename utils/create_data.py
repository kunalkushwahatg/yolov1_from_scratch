import numpy as np
import os
from tqdm import tqdm_gui
from get_labels import Label
import pickle 

images_path = "E:/VOC2012_train_val/VOC2012_train_val/JPEGImages/"
annotations_path = "E:/VOC2012_train_val/VOC2012_train_val/Annotations/"

dataset = []

for i,image_path in tqdm_gui(enumerate(os.listdir(images_path))):

    annotation_filename = image_path.split(".")[0]+".xml"
    image_path = images_path+image_path
    annotation_path = annotations_path+annotation_filename
    gl = Label(n_grids=7,n_boxes=2)  
    try:
        image , label = gl.get_label(image_path,annotation_path)
        dataset.append([image,label])
    except Exception as e:
        pass
    
pickle.dump(dataset,open("dataset1.pkl","wb"))