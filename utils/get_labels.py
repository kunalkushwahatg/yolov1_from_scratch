import numpy as np
import xml.etree.ElementTree as ET
from helper_functions import read_annotation , reduce_image_size
import pickle 

class Label():
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

    def get_annotations(self,label,image_size,threshold_confiendence):
        
        matrix = np.argmax( label,axis=2)
        flat_indices = np.flatnonzero(matrix)
        row_indices, col_indices = np.unravel_index(flat_indices,matrix.shape)
        positions = list(zip(row_indices, col_indices))

        annotaions = []
        for i,(grid_y,grid_x) in enumerate(positions):
            if annotaions >= threshold_confiendence:
                cell_x, cell_y, box_width, box_height, confiedence_score  =  label[grid_y, grid_x,5*i:5*(i+1)]
                x = (grid_x + cell_x) / self.n_grids
                y = (grid_y + cell_y) / self.n_grids
                X = x * image_size[1]
                Y = y * image_size[0]
                annotaions.append([X,Y,box_width,box_height,confiedence_score])
        return annotaions


    
f = open("C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/dataset1.pickle","rb")
dataset = pickle.load(f)
