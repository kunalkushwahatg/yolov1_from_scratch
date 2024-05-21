import cv2
import xml.etree.ElementTree as ET
import numpy as np
from helper_functions import read_annotation
from get_labels import GetLabel

xml_filename = "C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/VOC2012_train_val/VOC2012_train_val/Annotations/2007_000033.xml"
img = cv2.imread("C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/VOC2012_train_val/VOC2012_train_val/JPEGImages/2007_000033.jpg")
bounding_box_properties = {"color":(0, 0, 255) ,"thickness": 2  }
new_width = 224 
new_height = 224
classes =['sofa', 'bus', 'cow', 'cat', 'dog', 'bird', 'sheep', 'motorbike', 'pottedplant', 'diningtable', 
           'chair', 'aeroplane', 'tvmonitor', 'train', 'bicycle', 'bottle', 'boat', 'person', 'horse', 'car']

S = 7  
B = 2 
C = 20 


def read_bounding_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        #((top left),(bottom right))
        boxes.append(((xmin, ymin), (xmax, ymax)) ) 


    size = root.findall('size')[0]
    width = int(size.find('width').text)
    heigth = int(size.find('height').text)
    image_size = (width , heigth)

    
    return boxes , image_size

coordinates , image_size = read_bounding_boxes(xml_filename)


img = cv2.resize(img,(new_width,new_height))

reduce_ratio_width = new_width/image_size[0]
reduce_ratio_height = new_height/image_size[1]

new_coordinates = []
for coordinate in coordinates:
    new_coordinates.append(((int(coordinate[0][0]*reduce_ratio_width) , int(coordinate[0][1]*reduce_ratio_height)),
                            (int(coordinate[1][0]*reduce_ratio_width) , int(coordinate[1][1]*reduce_ratio_height))))
for coordinate in new_coordinates:


    cv2.rectangle(img, coordinate[0], coordinate[1], bounding_box_properties["color"], bounding_box_properties["thickness"])
    box_width = coordinate[1][0] - coordinate[0][0]
    box_height = coordinate[1][1] - coordinate[0][1] 


    center_coordinates = ( int((box_width/2) + coordinate[0][0]) ,  int((box_height/2) + coordinate[0][1]) )
    cv2.circle(img,center=center_coordinates,radius=2,color=(0,225,0), thickness=3)



def draw_grid():
    for i in range(S-1):
        cv2.line(img,pt1=(0,int((new_height/7))*(i+1)),pt2=(new_height,int((new_height/7))*(i+1)),color=(0,225,0), thickness=1)
        cv2.line(img,pt1=(int((new_width/7))*(i+1),0),pt2=(int((new_width/7))*(i+1),new_height),color=(0,225,0), thickness=1)


def create_labels():
    labels = np.zeros((S,S, B*5 + B*C))

    for i,coordinate in enumerate(new_coordinates):

        box_width = coordinate[1][0] - coordinate[0][0]
        box_height = coordinate[1][1] - coordinate[0][1] 

        X , Y  =  int((box_width/2) + coordinate[0][0]) ,  int((box_height/2) + coordinate[1][1]) 

        x , y = X/new_width , Y/new_height

        grid_x = int(x * S)
        grid_y = int(y * S)
        cell_x = (x * S) - grid_x
        cell_y = (y * S) - grid_y

        box = np.array([cell_x, cell_y, box_width, box_height, 1.0])
        labels[grid_y, grid_x, :5] = box
        labels[grid_y, grid_x, ( (i+1)*10) + 5] = 1
        return labels , grid_x  , grid_y


create_labels()
cv2.imshow('Image with Rectangle', img)
cv2.waitKey(0)  # 0 means wait indefinitely
cv2.destroyAllWindows()


