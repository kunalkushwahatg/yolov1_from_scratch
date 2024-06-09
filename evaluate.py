 #recives the batch of dataloader and computes the accuracy and loss use class
import config
import torch
from torch.functional import F
import matplotlib.pyplot as plt
class Evaluate:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        pass

    
    def batch_loss(self,model,criteriation,data):
        '''
        input: model : model object
               criteriation : loss function object
               data : batch of data in formate [image,label] 
        output: loss : loss of the batch
        '''
        model.eval()
        with torch.no_grad():
            output = model(data[0].to(config.DEVICE))
            loss = criteriation(output,data[1].to(config.DEVICE))
        return loss
    
    
    def dataloader_loss(self,model,criteriation,dataloader):
        '''
        input: model : model object
               criteriation : loss function object
               dataloader : dataloader object
        output: loss : loss of the dataloader
        '''
        loss = 0
        for batch in dataloader:
            loss += self.batch_loss(model,criteriation,batch)
        return loss/len(dataloader)
    
    def plot_loss(self,loss,label,save=None):
        '''
        input: loss : list of loss
                label : label for the plot
                save : path to save the plot
        output: Saves the plot and returns None
        '''
        
        plt.plot(loss,label=label)
        if save:
            try:
                plt.savefig(save)
            except:
                print('Invalid path')

        plt.legend()
        plt.show()