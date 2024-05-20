import cv2
import xml.etree.ElementTree as ET

xml_filename = "C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/2007_000063.xml"
img = cv2.imread("C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/2007_000063.jpg")
bounding_box_properties = {"color":(0, 0, 255) ,"thickness": 2  }

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

new_width = 224 
new_height = 224 

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
    box_height = coordinate[0][1] - coordinate[1][1]
    center_coordinates = ( int((box_width/2) + coordinate[0][0]) ,  int((box_height/2) + coordinate[1][1]) )


cv2.imshow('Image', img)
cv2.waitKey(0) 
cv2.destroyAllWindows()

