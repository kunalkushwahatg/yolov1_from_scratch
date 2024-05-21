import numpy as np
import xml.etree.ElementTree as ET
from helper_functions import read_annotation , reduce_image_size

class GetLabel():
    def __init__(self,n_grids,n_boxes):
        self.n_grids = n_grids
        self.n_boxes = n_boxes
        self.classes = ['sofa', 'bus', 'cow', 'cat', 'dog', 'bird', 'sheep', 'motorbike', 'pottedplant', 'diningtable', 
                        'chair', 'aeroplane', 'tvmonitor', 'train', 'bicycle', 'bottle', 'boat', 'person', 'horse', 'car']

        self.class_size = len(self.classes)
    
    def get_label(self,image_path,annotation_path,new_size = (224,224)):


        coordinates , names = read_annotation(annotation_path,self.n_boxes)
        image , new_coordinates = reduce_image_size(image_path,new_size,coordinates)
        
        labels = np.zeros((self.n_grids,self.n_grids, self.n_boxes*5 + self.n_boxes*self.class_size ))



        for i,coordinate in enumerate(new_coordinates):

            n_boxes = len(new_coordinates)

            class_id = self.classes.index(names[i])

            box_width = coordinate[1][0] - coordinate[0][0]
            box_height = coordinate[1][1] - coordinate[0][1] 


            


            X , Y  =  int((box_width/2) + coordinate[0][0]) ,  int((box_height/2) + coordinate[0][1])

            x , y = X/new_size[0] , Y/new_size[1]


            grid_x = int(x * self.n_grids)
            grid_y = int(y * self.n_grids)
            cell_x = (x * self.n_grids) - grid_x
            cell_y = (y * self.n_grids) - grid_y
    
            box = np.array([cell_x, cell_y, box_width, box_height, 1.0])


            labels[grid_y, grid_x,5*i:5*(i+1)] = box
            labels[grid_y, grid_x, ( 5*n_boxes + i*self.class_size) + class_id] = 1
        return np.array(image),labels

