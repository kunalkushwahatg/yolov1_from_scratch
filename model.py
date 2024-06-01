import torch.nn as nn
import torch
import config
from torch.nn.modules.flatten import Flatten
class YoloMain(nn.Module):
    def __init__(self):
        super(YoloMain, self).__init__()

        # Define the layers for the model
        
        #convolutional layer 7X7X64-s-2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.leaky_relu1 = nn.LeakyReLU()  
        #max pooling layer 2X2-s-2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 3X3X192 and max pooling layer 2X2-s-2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU()  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 1X1X128
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1)
        self.leaky_relu3 = nn.LeakyReLU()  

        #convolutional layer 3X3X256
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.leaky_relu4 = nn.LeakyReLU()  

        #convolutional layer 1X1X256
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.leaky_relu5 = nn.LeakyReLU()  

        #convolutional layer 3X3X512 and max pooling layer 2X2-s-2
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu6 = nn.LeakyReLU()  
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 1X1X256 4 times
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.leaky_relu7 = nn.LeakyReLU()  
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.leaky_relu8 = nn.LeakyReLU()  
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.leaky_relu9 = nn.LeakyReLU()  
        
        #convolutional layer 3X3X512 4 times
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu10 = nn.LeakyReLU()  
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu11 = nn.LeakyReLU()  
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu12 = nn.LeakyReLU()  
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu13 = nn.LeakyReLU()  

        #max pooling layer 2X2-s-2
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 1X1X512 2 times
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.leaky_relu14 = nn.LeakyReLU()  
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.leaky_relu15 = nn.LeakyReLU()  

        #convolutional layer 3X3X1024 2 times
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.leaky_relu16 = nn.LeakyReLU()  
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.leaky_relu17 = nn.LeakyReLU()  

        #convolutional layer 3X3X1024-s-2
        self.conv18 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2)
        self.leaky_relu18 = nn.LeakyReLU()  

        #convolutional layer 3X3X1024
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.leaky_relu19 = nn.LeakyReLU()  

        #flatten the output
        self.flatten = Flatten()

        #fully connected linear layer
        self.fc1 = nn.Linear(3*3*1024, 4096)
        self.leaky_relu20 = nn.LeakyReLU()  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, config.N_GRIDS * config.N_GRIDS * (config.N_BOXES * 5 + config.N_BOXES * config.N_CLASSES))

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.conv5(x)
        x = self.leaky_relu5(x)
        x = self.conv6(x)
        x = self.leaky_relu6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.leaky_relu7(x)
        x = self.conv8(x)
        x = self.leaky_relu8(x)
        x = self.conv9(x)
        x = self.leaky_relu9(x)
        x = self.conv10(x)
        x = self.leaky_relu10(x)
        x = self.conv11(x)
        x = self.leaky_relu11(x)
        x = self.conv12(x)
        x = self.leaky_relu12(x)
        x = self.conv13(x)
        x = self.leaky_relu13(x)
        x = self.pool4(x)
        x = self.conv14(x)
        x = self.leaky_relu14(x)
        x = self.conv15(x)
        x = self.leaky_relu15(x)
        x = self.conv16(x)
        x = self.leaky_relu16(x)
        x = self.conv17(x)
        x = self.leaky_relu17(x)
        x = self.conv18(x)
        x = self.leaky_relu18(x)
        x = self.conv19(x)
        x = self.leaky_relu19(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_relu20(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, config.N_GRIDS, config.N_GRIDS, (config.N_BOXES * 5 + config.N_BOXES * config.N_CLASSES))
        return x

#image_shape = (batch_size,_num_channels, height, width)
x = torch.randn(1, 3, 224, 224,device=config.DEVICE)   
model = YoloMain()
model = model.to(config.DEVICE)

#print runtime in sec of model
import time
start = time.time()
output = model(x)
end = time.time()

#calculate fps
fps = 1/(end-start)


print('FPS:',fps)
print('Runtime:', end - start , 'sec')
#print modelsize in MB
print('Model size:',sum(p.numel() for p in model.parameters()) / 1e6 , 'MB')