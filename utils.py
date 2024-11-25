import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from tqdm import tqdm
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).squeeze().long()
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(outputs.argmax(1).flatten().cpu().numpy())
        all_labels.extend(labels.flatten().cpu().numpy())

    # Calculate metrics
    # accuracy = accuracy_score(all_labels, all_preds)
    # precision = precision_score(all_labels, all_preds, average='binary')
    # recall = recall_score(all_labels, all_preds, average='binary')
    # f1 = f1_score(all_labels, all_preds, average='binary')

    return sum(losses)/len(losses)#, accuracy, precision, recall, f1

def validate(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).squeeze().long()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            all_preds.extend(outputs.argmax(1).flatten().cpu().numpy())
            all_labels.extend(labels.flatten().cpu().numpy())

    # Calculate metrics
    # accuracy = accuracy_score(all_labels, all_preds)
    # precision = precision_score(all_labels, all_preds, average='binary')
    # recall = recall_score(all_labels, all_preds, average='binary')
    # f1 = f1_score(all_labels, all_preds, average='binary')

    return sum(losses)/len(losses)#, accuracy, precision, recall, f1

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes