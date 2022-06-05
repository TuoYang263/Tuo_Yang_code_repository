from info_extraction import *
from MyDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from train import *
from plot_use_plt import *
from Resnet import *
from plot_confusion_matrix import *
import torch.optim as optim

if __name__ == '__main__':

    # building ground truth files
    info_extraction()

    # backup transformer options

    # create transformers for three datasets respectively
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    validate_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # create DataSet objects for two datasets
    train_data = MyDataset(txt=root+'train.txt', transform=train_transforms)
    validation_data = MyDataset(txt=root+'validation.txt', transform=validate_transforms)

    # create DataLoaders for two datasets
    train_loader = DataLoader(dataset=train_data, batch_size=24, shuffle=True, num_workers=1)

    validate_loader = DataLoader(dataset=validation_data, batch_size=24, shuffle=True, num_workers=1)

    # get the network model by calling function
    # we got 41 classes need to be classified
    # 1.transfer learning
    net = resnet50(pretrained=True)
    print(net.buffers)
    # replace the last layer
    only_train_fc = True
    if only_train_fc:
        for param in net.parameters():
            param.requires_grad_(False)
    # only 41 classes here
    net.fc = torch.nn.Linear(in_features=2048, out_features=41, bias=True)
    # check if parameters have been changed print(net.buffers)

    learning_rate = 0.009

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.8)

    # specify device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # copy all read tensor variables from the start to device,all calculation will
    # be done on GPU
    net.to(device)

    num_epoch = 30

    global start_epoch
    start_epoch = 0
    show_mode = 0  # when it was set as 0, plt plotting, set as 1, txt log plotting

    mean_loss_list, train_acc_list, test_acc_list = train(net, train_loader,
                                                          validate_loader, start_epoch=start_epoch, optimizer=optimizer,
                                                          device=device, num_epochs=num_epoch)

    plot_use_plt(mean_loss_list, train_acc_list, test_acc_list, num_epoch)









