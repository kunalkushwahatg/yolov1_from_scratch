import os
from tqdm import tqdm_gui
from get_labels import Label
import pickle 
import argparse

parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument('--files', nargs='+', help='List of file paths to process')

args = parser.parse_args()


images_path = args.files[0]
annotation_path = args.files[1]

dataset = []

for i,image_path in tqdm_gui(enumerate(os.listdir(images_path))):

    annotation_filename = image_path.split(".")[0]+".xml"
    image_path = images_path+image_path
    annotation_path = annotation_path+annotation_filename
    gl = Label(n_grids=7,n_boxes=2)  
    try:
        image , label = gl.get_label(image_path,annotation_path)
        dataset.append([image,label])
    except Exception as e:
        pass
    
pickle.dump(dataset,open("dataset2.pkl","wb"))