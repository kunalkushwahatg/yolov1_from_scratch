import cv2
import config
from utils.helper_functions import read_annotation , reduce_image_size
from PIL import Image
import numpy as np

class Visualize():
    def __init__(self):
        self.bounding_box_properties = {"color":(0, 0, 255) ,"thickness": 2  }
        self.new_width = config.WIDTH
        self.new_height = config.HEIGHT
        self.classes  =['sofa', 'bus', 'cow', 'cat', 'dog', 'bird', 'sheep', 'motorbike', 'pottedplant', 'diningtable', 
           'chair', 'aeroplane', 'tvmonitor', 'train', 'bicycle', 'bottle', 'boat', 'person', 'horse', 'car']


    def draw_bounding_boxes(self,image,coordinates):
        '''
        image : image on which bounding boxes are to be drawn in numpy array format
        coordinates : list of tuples containing coordinates of bounding boxes 
                      in the format [((top left),(bottom right)),other box]
        '''

        for coordinate in coordinates:
            cv2.rectangle(image, coordinate[0], coordinate[1], self.bounding_box_properties["color"], self.bounding_box_properties["thickness"])
        return image
    
    def draw_grid(self,image):
        '''
        image : image on which grid is to be drawn in numpy array format
        '''
        for i in range(config.S-1):
            cv2.line(image,pt1=(0,int((self.new_height/config.S))*(i+1)),pt2=(self.new_height,int((self.new_height/config.S))*(i+1)),color=(0,225,0), thickness=1)
            cv2.line(image,pt1=(int((self.new_width/config.S))*(i+1),0),pt2=(int((self.new_width/config.S))*(i+1),self.new_height),color=(0,225,0), thickness=1)
        return image
    
    def read_image(self,path):
        image  = Image.open(path)
        #convert PIL image to cv2 image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    
    def draw_box_from_file(self,image_path,xml_path):
        '''
        image_path : path of the image
        xml_path : path of the xml file
        '''
        coordinates , _ = read_annotation(xml_path,config.N_BOXES)      
        image = self.read_image(str(image_path))
        image , new_coordinates = reduce_image_size(image,(self.new_width,self.new_height),coordinates)
        image = cv2.resize(image,(self.new_width,self.new_height))
        
        print(new_coordinates)
        image = self.draw_bounding_boxes(image,new_coordinates)
        cv2.imshow('Image with Rectangle', image)
        cv2.waitKey(0)




