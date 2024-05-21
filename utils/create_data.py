import numpy as np
import os
from tqdm import tqdm_gui
from get_labels import GetLabel
import pickle 

images_path = "C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/VOC2012_train_val/VOC2012_train_val/JPEGImages/"
annotations_path = "C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/VOC2012_train_val/VOC2012_train_val/Annotations/"

dataset = []

for i,image_path in tqdm_gui(enumerate(os.listdir(images_path))):

    annotation_filename = image_path.split(".")[0]+".xml"
    image_path = images_path+image_path
    annotation_path = annotations_path+annotation_filename
    gl = GetLabel(n_grids=7,n_boxes=2)  
    try:
        image , label = gl.get_label(image_path,annotation_path)
    except Exception as e:
        pass
    dataset.append([image,label])



with open("C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/dataset.pickle","wb") as f:
    pickle.dump(dataset,f)
