#%%
import os
import copy
import time

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import pandas as pd


#%%
class SubsetDataset(Dataset):
    r"""
    Converts ImageFolder Dataset Subset into Dataset
    https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)


#%%
trainpath = 'd:/shopee/pyc2/train/train/train'


def load_split_train(trainpath):
    # load image into ImageFolder
    imagefolder = datasets.ImageFolder(trainpath)
    # random split ImageFolder into train and validate
    lengths = [int(len(imagefolder)*0.9), int(len(imagefolder)*0.1)]
    train_set, val_set = torch.utils.data.dataset.random_split(imagefolder, lengths)

    # transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = SubsetDataset(train_set, transform=train_transform)
    val_set = SubsetDataset(val_set, transform=val_transform)

    print(len(imagefolder))
    print(len(train_set))
    print(len(val_set))

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=True)

    print(len(train_dataloader))
    print(len(val_dataloader))

    # train, val, test, using enumerate
    return [train_dataloader, val_dataloader]


#%%
class InferenceDataset(Dataset):
    """
    Derived from torchvision.datasets.folder.DatasetFolder
    """

    def __init__(self, inferpath, transform=None):
        self.transform = transform
        self.inferpath = inferpath
        self.samples = []

        # make_dataset
        fnames = sorted(os.listdir(inferpath))
        for fname in fnames:
            self.samples.append(fname)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            triple: (image, label, path) where target is class_index
        """
        path = self.samples[index]
        abspath = os.path.join(self.inferpath, path)
        sample = Image.open(abspath).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path


#%%
inferpath = 'd:/shopee/pyc2/test/test/test'


def load_infer(inferpath):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    inferdatasest = InferenceDataset(inferpath, transform=transform)

    infer_dataloader = torch.utils.data.DataLoader(inferdatasest, batch_size=256)

    return infer_dataloader

#%%
# pylint:disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint:enable=E1101

#%%
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
# setting device on GPU if available, else CPU
def print_gpu_usage():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


#%%
# autosave the best model
def autosave_best(epoch, model, optimizer):
    state = {
        'epoch':epoch,
        'state_dict':model_ft.state_dict(),
        'optimizer':optimizer.state_dict()
    }
    torch.save(state, 'd:/shopee/pyc2/model_c2v21_%s.pth' % epoch)
    print('Epoch {} saved'.format(epoch))


#%%
# auto-log history
def autolog(epoch, history):
    # track accuracy, loss
    with open('d:/shopee/pyc2/model_c2v21_hist.txt', 'a+') as hist_file:
        hist_file.write('{}\n'.format(epoch))
        for phase in ['train', 'val']:
            hist_file.write( '{} Loss: {:f} Acc: {:f}\n'.format( phase, history[phase]['loss'][-1], history[phase]['acc'][-1] ) )
        hist_file.write('\n')

#%%
def print_time_elapsed(since):
    time_elapsed = time.time() - since
    print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# %%
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, dataloaders, criterion, optimizer, start_epoch=0, num_epochs=1):
    since = time.time()

    history = {'train': {'loss': [], 'acc': []}, 'val': {'loss': [], 'acc': []}}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(start_epoch+epoch, start_epoch+num_epochs-1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for counter, phase in enumerate(['train', 'val']):
            if phase == 'train':
                # model to training mode
                model.train()
            else:
                # model to evaluate mode
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            progress = 0
            # iterate over data
            for inputs, labels in dataloaders[counter]:
                progress = progress + 1
                # total 371 + 42
                if progress % 40 == 0 or progress == 1 or progress == 2:
                    print(progress)
                    print_time_elapsed(since)
                    #print_gpu_usage()
                    #print()

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero param gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # preds = prediction
                    # pylint:disable=E1101
                    _, preds = torch.max(outputs, 1)
                    # pylint:enable=E1101

                    # backward + optimize if only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #endwith

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # pylint:disable=E1101
                running_corrects += torch.sum(preds == labels.data)
                # pylint:enable=E1101
            #endforinputlabel

            epoch_loss = running_loss / len(dataloaders[counter].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[counter].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print()

            # deep copy model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                autosave_best(start_epoch+epoch, model, optimizer)

            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)
        #endforphase

        autolog(start_epoch+epoch, history)
    #endforepoch

    print_time_elapsed(since)
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
#endfunction


#%%
def infer_model(model, dataloaders):
    model.eval()

    since = time.time()

    path = []
    pred = []

    with torch.no_grad():
        progress = 0
        for inputs, paths in dataloaders:
            progress = progress + 1
            if progress % 40 == 0 or progress == 1 or progress == 2:
                print(progress)
                print_time_elapsed(since)
                #print_gpu_usage()
                #print()

            inputs = inputs.to(device)

            outputs = model(inputs)
            # pylint:disable=E1101
            _, preds = torch.max(outputs, 1)
            # pylint:enable=E1101

            path += paths
            pred += preds

    df = pd.DataFrame()
    df = pd.DataFrame({'filename': path, 'category': pred})

    print('Eval Complete')

    return df


#%%
def initialize_restnet18(resume=False, statepath=None):
    # initialize model
    model_ft = models.resnet18(pretrained=True)
    # in_feaures, num_classes = 42
    model_ft.fc = nn.Linear(model_ft.fc.in_features, 42)
    # recommended move model to GPU before optimizer
    # https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165
    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    start_epoch = 0

    # load model
    if resume:
        state = torch.load(statepath)
        start_epoch = state['epoch']+1
        model_ft.load_state_dict(state['state_dict'])
        model_ft = model_ft.to(device)
        optimizer.load_state_dict(state['optimizer'])

    '''
    # show optimized params
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    print(model_ft)
    '''

    # setup loss fxn
    criterion = nn.CrossEntropyLoss()

    return model_ft, criterion, optimizer, start_epoch


#%%
'''
dataloaders = load_split_train(trainpath)

model_ft, criterion, optimizer, start_epoch = initialize_restnet18(resume=True, statepath='d:/shopee/pyc2/model_c2v21_22.pth')
# train and validate
model_ft, _ = train_model(model_ft, dataloaders, criterion, optimizer, start_epoch, 2)
'''
#%%

dataloaders = load_infer(inferpath)
model_ft, _, _, _ = initialize_restnet18(resume=True, statepath='d:/shopee/pyc2/model_c2v21_22.pth')
# evaluate/ test
df = infer_model(model_ft, dataloaders)

# %%

#change tensor object to fundamental type
df['category'] = df['category'].map(lambda x: x.item())
# export
df.to_csv('d:/shopee/pyc2/out.csv', index=False)
print(df)

# %%
