#reads the pickle file 
import pickle
from utils.dataset import MyDataset
from torch.utils.data import random_split 
from torch.utils.data import DataLoader
import config 
from model import YoloMain
from yolo_loss import YoloLoss
import torch
import torch.optim as optim
from evaluate import Evaluate
from PIL import Image


class Train:
    def __init__(self):
        self.batch_size = config.BATCH_SIZE
        self.transfrom = config.TRANSFORM
        self.epochs = config.EPOCHS
        self.device  = config.DEVICE
        self.lr = config.LEARING_RATE
        self.weight_decay = config.WEIGHT_DECAY
        self.momentum = config.MOMENTUM
        self.print_every =  config.PRINT_EVERY

    
    def train(self,model,train_loader,val_loader,path_for_gif=None):
        model = model.to(device=self.device)
        criteriation = YoloLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        eval = Evaluate()


        for epoch in range(self.epochs):
            # Clear the GPU cache at the end of each epoch
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            model.train()
            for batch in train_loader:

                # Get the image data and label from the batch
                img_data, label = batch

                # Forward pass
                output = model(img_data.to(self.device))

                # Compute the loss
                loss = criteriation(output, label.to(self.device))

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Zero the gradients
                optimizer.zero_grad()
            
            if path_for_gif:
                eval.images_for_gif.append(Image.open(path_for_gif))


            if epoch % self.print_every == 0:
                train_loss = eval.dataloader_loss(model,criteriation,train_loader)
                val_loss = eval.dataloader_loss(model,criteriation,val_loader)
                print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
                eval.train_loss.append(train_loss)
                eval.val_loss.append(val_loss)

        if path_for_gif:
            eval.save_gif(save_path=path_for_gif)

        return model,eval.train_loss,eval.val_loss


#data is in formate [image,label]
data = pickle.load(open('C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/dataset1.pickle', 'rb'))  
#create a dataset class of pytorch 
dataset = MyDataset(data,transform=transfrom)

#split the dataset for train and val 
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#creates the dataloader for train and validation 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_laoder = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)






        


