from torchvision import transforms

#creates neccessary config parameters 

WIDTH = 224
HEIGHT = 224

N_GRIDS = 7
N_BOXES = 2
N_CLASSES = 20
IMAGE_SIZE = 224
DEVICE = 'cuda'
BATCH_SIZE = 64
EPOCHS = 100
LEARING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
PRINT_EVERY = 5


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

