    
import torch.nn as nn
import config
import torch 
#incomplete
class YoloFast(nn.Module):
    #We also train a fast version of YOLO designed to push the boundaries of fast object detection. Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 24) and fewer filters in those layers. Other than the size of the network, all training and testing parameters are the same between YOLO and Fast YOLO
    def __init__(self):
        super(YoloFast, self).__init__()

        # Define the layers for the model
        
        #convolutional layer 7X7X64-s-2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        #max pooling layer 2X2-s-2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 3X3X192 and max pooling layer 2X2-s-2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 1X1X128
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1)

        #convolutional layer 3X3X256
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        #convolutional layer 1X1X256
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)

        #convolutional layer 3X3X512 and max pooling layer 2X2-s-2
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #convolutional layer 1X1X256 2 times
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)

        #convolutional layer 3X3X512 4 times
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        #define conv layers make size to 1024x3x3
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x
    
x = torch.randn(1, 3, 224, 224)
model = YoloFast()
output = model(x)


#image_shape = (batch_size,_num_channels, height, width)
x = torch.randn(1, 3, 224, 224,device=config.DEVICE)   
model = YoloFast()
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