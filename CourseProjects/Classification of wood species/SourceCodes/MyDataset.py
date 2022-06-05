from torch.utils.data import Dataset,DataLoader
from PIL import Image


# the address of ground truth info
root = './labels/'


# define the format of reading files
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # initialize some parameters and call datasets
    def __init__(self, txt, transform=None,
                 target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []
        # use the read-only way to open the txt by incoming path and txt text parameter
        fh = open(txt, 'r')
        for line in fh:
            # make iteration for this list, check every row of txt text file
            line = line.strip('\r\n')
            line = line.rstrip('\r\n')
            # use split function to split this row into lists
            # its default parameter is space
            words = line.split()
            # store contents inside txt into list imgs
            # word[0] is the content of image, words[1] is label
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # do some data preprocessing to data and return required info
    # it's necessary and used for reading every element's info by indices
    # the info we can acquire from circularly reading every batch depends on
    # contents returned from this function
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # read RGB bands info from image according to location inside the label
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # function used for returning the length of datasets
    def __len__(self):
        return len(self.imgs)








