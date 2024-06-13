 #recives the batch of dataloader and computes the accuracy and loss use class
import config
import torch
import matplotlib.pyplot as plt
from get_labels import Label
from visualization import Visualize
import imageio


class Evaluate:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.images_for_gif = []

    
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

    
    def predict(self,model,img_data):
        '''
        input: model : model object
               data : image data
        output: prediction : model prediction
        '''

        model.eval()
        with torch.no_grad():
            prediction = model(img_data.to(config.DEVICE))
            return prediction
        
    def predict_image(self,model,img_data):
        '''
        input: model : model object
               data : image data
        output: prediction : bounding box in image 
        '''
        prediction = self.predict(model,img_data)
        label = Label()
        visualize = Visualize()

        coordinates = label.get_annotations(prediction)
        image = visualize.draw_bounding_boxes(image,coordinates)
        return  image
    
    def save_gif(self,save_path):
        '''
        input: images :save_path : path to save the gif
        output: Saves the gif
        '''
        # Convert BGR images to RGB
        # Save images as gif
        imageio.mimsave(save_path, self.images_for_gif, duration=0.5)  # duration is the time between frames in seconds

