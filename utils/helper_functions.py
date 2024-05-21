import cv2
import xml.etree.ElementTree as ET
import numpy as np
from lxml import etree


def read_annotation(xml_path,n_boxes):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    names = []
    for i,obj in enumerate(root.findall('object')):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        name = obj.find('name').text
        names.append(name)
        #((top left),(bottom right))
        boxes.append(((xmin, ymin), (xmax, ymax)) ) 
        if n_boxes <= i+1 :
            break
    return boxes,names


def reduce_image_size(image_path,new_size,box_coordinates):
    image = cv2.imread(image_path)

    image_size = image.shape

    new_width = new_size[0]
    new_height = new_size[1]

    image = cv2.resize(image,(new_width,new_height))

    reduce_ratio_width = new_width/image_size[1]
    reduce_ratio_height = new_height/image_size[0]

    new_coordinates = []
    for coordinate in box_coordinates:
        new_coordinates.append(((int(coordinate[0][0]*reduce_ratio_width) , int(coordinate[0][1]*reduce_ratio_height)),
                            (int(coordinate[1][0]*reduce_ratio_width) , int(coordinate[1][1]*reduce_ratio_height))))
    return image , new_coordinates

