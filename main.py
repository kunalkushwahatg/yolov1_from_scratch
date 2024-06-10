import torch
from torch.utils.data import DataLoader, random_split
from train import Train  
from utils.dataset import MyDataset
import pickle
import config
from model import YoloMain

def main():
    
    # Load data
    data = pickle.load(open('C:/Users/kunalkushwahatg/Desktop/yolov1_from_scratch/data/dataset1.pickle', 'rb'))

    # Create a dataset
    dataset = MyDataset(data, transform=config.TRANSFORM)

    # Split the dataset for training and validation
    train_size = int(config.TRAIN_TEST_SPLIT_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    
    model = YoloMain()  

    # Create a Train object and start training
    trainer = Train(model, train_loader, val_loader)
    model, train_loss, val_loss = trainer.train(config.EPOCHS)

    # Save the trained model
    torch.save(model.state_dict(), 'models/model.pth')
    print("model successfully saved ")
    print("Training Loss:", train_loss)
    print("Validation Loss:", val_loss)

if __name__ == "__main__":
    main()