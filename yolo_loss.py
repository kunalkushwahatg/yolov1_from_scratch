import torch
from config import DEVICE,BATCH_SIZE
import torch.nn as nn

'''
This is the loss function for the YOLO model. The loss function is a combination of 4 losses:
1. Coordination loss: This loss is calculated as the sum of the squared difference between the predicted and actual bounding box coordinates.
2. Confidence loss: This loss is calculated as the sum of the squared difference between the predicted and actual confidence scores.
3. No object loss: This loss is calculated as the sum of the squared difference between the predicted and actual confidence scores for grid cells that do not contain an object.
4. Class loss: This loss is calculated as the sum of the squared difference between the predicted and actual class probabilities.
'''


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):

        '''
        Input:
                Predictions: The output of the YOLO model. It is a tensor of shape (batch_size, S, S, Bx5 + C), where S is the number of grid cells, B is the number of bounding boxes per grid cell, and C is the number of classes.
                Targets: The ground truth labels. It is a tensor of shape (batch_size, S, S, 5 + C), where the first 5 elements are the bounding box coordinates and confidence score, and the remaining elements are the class probabilities.
        Output:
                Loss: The total loss for the YOLO model.
        '''

        coord_mask1 = targets[..., 4].unsqueeze(-1)
        coord_mask2 = targets[..., 9].unsqueeze(-1)


        coord_loss1 = self.lambda_coord * torch.sum(coord_mask1 * (
            (targets[..., :2] - predictions[..., :2]) ** 2 +
            (torch.sqrt(targets[..., 2:4] + 1e-6) - torch.sqrt(targets[..., 2:4] + 1e-6)) ** 2
        ))
        coord_loss2 = self.lambda_coord * torch.sum(coord_mask2 * (
            (targets[...,5:7] - predictions[..., 5:7]) ** 2 +
            (torch.sqrt(targets[..., 7:9] + 1e-6) - torch.sqrt(targets[..., 7:9] + 1e-6)) ** 2
        ))

        coord_loss = (coord_loss1 + coord_loss2) * self.lambda_coord
        conf_loss1 = torch.sum( ((targets[..., 4] - predictions[..., 4]) ** 2).unsqueeze(-1) * coord_mask1)
        conf_loss2 = torch.sum(((targets[..., 9] - predictions[..., 9]) ** 2).unsqueeze(-1) * coord_mask2)

        conf_loss = conf_loss1 + conf_loss2

        noobj_loss1 = self.lambda_noobj * torch.sum((1 - coord_mask1) * (predictions[..., 4] ** 2).unsqueeze(-1))
        noobj_loss2 = self.lambda_noobj * torch.sum((1 - coord_mask2) * (predictions[..., 9] ** 2).unsqueeze(-1))

        noobj_loss = (noobj_loss1 + noobj_loss2) * self.lambda_noobj



        class_loss1 = torch.sum(coord_mask1 * ((targets[...,10:30] - predictions[...,10:30]) ** 2))
        class_loss2 = torch.sum(coord_mask2 * ((targets[...,30:] - predictions[...,30:]) ** 2))

        class_loss = class_loss1 + class_loss2

        loss = (coord_loss + conf_loss + noobj_loss + class_loss) / BATCH_SIZE

        return loss