#reads the pickle file 
from torch.utils.data import random_split 
import config 
from yolo_loss import YoloLoss
import torch
import torch.optim as optim
from evaluate import Evaluate
from PIL import Image
from tqdm.notebook import tqdm


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
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        eval = Evaluate()


        for epoch in range(self.epochs):
            # Clear the GPU cache at the end of each epoch
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            model.train()
            for batch in tqdm(train_loader):

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








        


