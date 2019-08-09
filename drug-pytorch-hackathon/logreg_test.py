import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data.sampler import SubsetRandomSampler
from Helper import hamming_score

from SIDER import *


def train(train_loader, model, criterion, optimizer, epoch):
    print("TRAIN", epoch)
    model.train()
    end = time.time()
    with torch.enable_grad():
        running_BCELoss = 0
        running_Hamming = 0
        for i, (X, y_labels, IDs) in enumerate(train_loader):
            y_labels = y_labels.float()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y_labels)
            running_BCELoss += loss.item()
            loss.backward()
            optimizer.step()
            running_Hamming += hamming_score(y_labels.detach().numpy(), outputs.detach().numpy())
            if i > 0:
                print("BCE:", running_BCELoss / i)
                print("Hamming:", running_Hamming / i)
            # if i % 20 == 19:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_BCELoss / i))
            #     running_BCELoss = 0


def evaluate(validation_loader, model, criterion, epoch):
    print("EVAL", epoch)
    model.eval()
    end = time.time()
    with torch.no_grad():
        running_BCELoss = 0
        running_Hamming = 0
        for i, (X, y_labels, IDs) in enumerate(validation_loader):
            y_labels = y_labels.float()
            outputs = model(X)
            loss = criterion(outputs, y_labels)
            running_BCELoss += loss.item()
            running_Hamming += hamming_score(y_labels.numpy(), outputs.numpy())
            if i > 0:
                print("BCE:", running_BCELoss / i)
                print("Hamming:", running_Hamming / i)



def main():
    all_data = SIDER(inputs_file="sider_latent.npy", outputs_file="SIDER_PTs.csv", IDs_file="sider_idxs_to_keep.npy")
    effects_dict = all_data.effects_dict

    batch_size = 8
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    learning_rate = 0.0001

    # Creating data indices for training and validation splits:
    dataset_size = len(all_data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(all_data, batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    validation_loader = torch.utils.data.DataLoader(all_data, batch_size=batch_size,
                                                    sampler=valid_sampler,
                                               num_workers=0)

    latent_dims = all_data.input_len
    num_classes = all_data.label_classes

    model = LogisticRegression(latent_dims, num_classes)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(train_loader, model, criterion, optimizer, 0)
    evaluate(validation_loader, model, criterion, 0)


if __name__ == "__main__":
    main()

