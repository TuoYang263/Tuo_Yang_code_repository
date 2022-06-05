from MyDataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from Resnet import *
from plot_confusion_matrix import *
from torch.autograd import Variable
import torch


if __name__ == '__main__':
    # specify device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # configure transformer for the test data
    test_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # create a dataset for test data
    test_data = MyDataset(txt=root+'test.txt', transform=test_transforms)

    # create a data loader for the test data
    test_loader = DataLoader(dataset=test_data, batch_size=24, shuffle=True, num_workers=1)

    # load trained network model
    # this model only save epochs,parameters and optimizer
    # we need another way to load models
    dict = torch.load('./trained_model/model.pt')
    # print(dict)
    # another way of loading models
    # load resnet50 model
    net = resnet50(pretrained=False, num_classes=41)

    # load the field 'model' inside the saved model
    net.load_state_dict(dict['model'])
    print(net)


    # create one empty matrix to store confusion matrix
    conf_matrix = torch.zeros(41, 41)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_images, batch_labels in test_loader:
            net.eval()
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            outputs = net(batch_images)

            _, prediction = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (prediction == batch_labels).sum()
            conf_matrix = confusion_matrix(prediction, labels=batch_labels, conf_matrix=conf_matrix)

    print('The accuracy of testing classification:%.3f%%' % (100 * correct / total))

    # conf_matrix is the numpy format
    # attack_types is the set of all class labels
    attack_types = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41']
    plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False,
                              title='Normalized confusion matrix')


