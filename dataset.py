from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image data and label from self.data
        img_data, label = self.data[index]

        # Convert the image data to a numpy ndarray
        img_data = np.array(img_data)

        # Convert the numpy ndarray to a PIL Image
        img = Image.fromarray(img_data)

        if self.transform:
            img = self.transform(img)

        return [img, label]