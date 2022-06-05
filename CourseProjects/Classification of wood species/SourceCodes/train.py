import torch
import torch.nn as nn
import time
from tqdm import tqdm

PATH = './trained_model/model.pt'
mean_loss_list = []
train_acc_list = []
test_acc_list = []


def test(net, test_iter, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for imgs, targets in test_iter:
            net.eval()
            y_hat = net(imgs.to(device)).argmax(dim=1)
            acc_sum += (y_hat == targets.to(device)).float().sum().cpu().item()
            net.train()
            n += targets.shape[0]
        return acc_sum/n


def train(net, train_iter, test_iter, start_epoch, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on:", device)
    # define loss function(common loss entropy)
    loss = torch.nn.CrossEntropyLoss()
    # the order of the batch 1st 2nd ...
    batch_count = 0
    # the length of training data
    nb = len(train_iter)
    for epoch in range(start_epoch, num_epochs):
        # start_epoch is used for loading unfinished training info last time
        train_loss_sum = 0.0   # loss value of training
        train_acc_sum = 0.0    # training accuracy
        n, start = 0, time.time()
        # use tqdm to observe the whole loading process of training dataset
        pbar = tqdm(enumerate(train_iter), total=nb)
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            # transfer pixel inside the network to get results
            y_hat = net(imgs)
            # calculate the loss value between the prediction results
            # and labels
            loss_value = loss(y_hat, targets)
            # clear the gradient
            optimizer.zero_grad()
            # back propagation
            loss_value.backward()
            # optimizer works
            optimizer.step()
            train_loss_sum += loss_value.cpu().item()
            # the reason why we use y_hat.argmax(dim=1) is because that
            # one vector containing 41 results is returned by this network
            # these 41 results represents the probability of their belonging class
            train_acc_sum += (y_hat.argmax(dim=1) == targets).sum().cpu().item()
            # take the maximum value's index as the result from 41 classes
            n += targets.shape[0]
            batch_count += 1
            s = '%g/%g  %g' % (epoch, num_epochs - 1, len(targets))
            # progress bar display
            pbar.set_description(s)

        mean_loss = train_loss_sum/batch_count
        train_acc = train_acc_sum/n
        test_acc = test(net, test_iter, device)
        # the next three lists as global variables used for plotting later
        mean_loss_list.append(mean_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('loss %.4f, train_acc %.3f, test_acc %.3f' % (mean_loss, train_acc, test_acc))
        # after all training done,create nodes list and save it to .pt file
        # which could save unfinished epoch as well
        chkpt = {'epoch': epoch,
                 'model': net.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(chkpt, PATH)
        del chkpt
    return mean_loss_list, train_acc_list, test_acc_list
