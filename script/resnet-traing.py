#  モデル作成
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# データの整理 ~ 学習
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class Bottleneck(nn.Module):
    """
    Bottleneckを使用したresidual blockクラス
    """
    def __init__(self, indim, outdim, is_first_resblock=False):
        super(Bottleneck, self).__init__()
        self.is_dim_changed = (indim != outdim)
        # W, Hを小さくしてCを増やす際はstrideを2にする +
        # projection shortcutを使う様にセット
        if self.is_dim_changed:
            if is_first_resblock:
                # 最初のresblockは(W､ H)は変更しないのでstrideは1にする
                stride = 1
            else:
                stride = 2
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=stride)
        else:
            stride = 1

        dim_inter = int(outdim / 4)
        self.conv1 = nn.Conv2d(indim, dim_inter , 1)
        self.bn1 = nn.BatchNorm2d(dim_inter)
        self.conv2 = nn.Conv2d(dim_inter, dim_inter, 3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(dim_inter)
        self.conv3 = nn.Conv2d(dim_inter, outdim, 1)
        self.bn3 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Projection shortcutの場合
        if self.is_dim_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    def __init__(self):

        super(ResNet50, self).__init__()

        # Due to memory limitation, images will be resized on-the-fly.
        self.upsampler = nn.Upsample(size=(224, 224))

        # Prior block
        self.layer_1 = nn.Conv2d(3, 64, 7, padding=3, stride=2)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # Residual blocks
        self.resblock1 = Bottleneck(64, 256, True)
        self.resblock2 = Bottleneck(256, 256)
        self.resblock3 = Bottleneck(256, 256)
        self.resblock4 = Bottleneck(256, 512)
        self.resblock5 = Bottleneck(512, 512)
        self.resblock6 = Bottleneck(512, 512)
        self.resblock7 = Bottleneck(512, 512)
        self.resblock8 = Bottleneck(512, 1024)
        self.resblock9 = Bottleneck(1024, 1024)
        self.resblock10 =Bottleneck(1024, 1024)
        self.resblock11 =Bottleneck(1024, 1024)
        self.resblock12 =Bottleneck(1024, 1024)
        self.resblock13 =Bottleneck(1024, 1024)
        self.resblock14 =Bottleneck(1024, 2048)
        self.resblock15 =Bottleneck(2048, 2048)
        self.resblock16 =Bottleneck(2048, 2048)

        # Postreior Block
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.upsampler(x)

        # Prior block
        x = self.relu(self.bn_1(self.layer_1(x)))
        x = self.pool(x)

        # Residual blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.resblock16(x)

        # Postreior Block
        x = self.glob_avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def load_data(path):
    """
    Load CIFAR10 data
    Reference:
      https://www.kaggle.com/vassiliskrikonis/cifar-10-analysis-with-a-neural-network/data

    """
    def _load_batch_file(batch_filename):
        filepath = os.path.join(path, batch_filename)
        unpickled = _unpickle(filepath)
        return unpickled

    def _unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin')
        return dict

    train_batch_1 = _load_batch_file('data_batch_1')
    train_batch_2 = _load_batch_file('data_batch_2')
    train_batch_3 = _load_batch_file('data_batch_3')
    train_batch_4 = _load_batch_file('data_batch_4')
    train_batch_5 = _load_batch_file('data_batch_5')
    test_batch = _load_batch_file('test_batch')

    num_classes = 10
    batches = [train_batch_1['data'], train_batch_2['data'], train_batch_3['data'], train_batch_4['data'], train_batch_5['data']]
    train_x = np.concatenate(batches)
    train_x = train_x.astype('float32') # this is necessary for the division below

    train_y = np.concatenate([np_utils.to_categorical(labels, num_classes) for labels in [train_batch_1['labels'], train_batch_2['labels'], train_batch_3['labels'], train_batch_4['labels'], train_batch_5['labels']]])
    test_x = test_batch['data'].astype('float32') #/ 255
    test_y = np_utils.to_categorical(test_batch['labels'], num_classes)
    print(num_classes)

    img_rows, img_cols = 32, 32
    channels = 3
    print(train_x.shape)
    train_x = train_x.reshape(len(train_x), channels, img_rows, img_cols)
    test_x = test_x.reshape(len(test_x), channels, img_rows, img_cols)
    train_x = train_x.transpose((0, 2, 3, 1))
    test_x = test_x.transpose((0, 2, 3, 1))
    per_pixel_mean = (train_x).mean(0) # 計算はするが使用しない

    train_x = [Image.fromarray(img.astype(np.uint8)) for img in train_x]
    test_x = [Image.fromarray(img.astype(np.uint8)) for img in test_x]

    train = [(x,np.argmax(y)) for x, y in zip(train_x, train_y)]
    test = [(x,np.argmax(y)) for x, y in zip(test_x, test_y)]
    return train, test, per_pixel_mean


class ImageDataset(Dataset):
    """
    データにtransformsを適用するためのクラス
    """
    def __init__(self, data, transform=None):

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# Googleドライブのマウント
# from google.colab import drive
# drive.mount('./drive')

BATCH_SIZE = 128
path = "./data/cifar-10-batches-py/"
train, test, per_pixel_mean = load_data(path)


# train dataの作成
train_transform = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: np.array(img)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.float()),
])
train_dataset = ImageDataset(train[:45000], transform=train_transform)
trainloader = DataLoader(train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)


# validation data, test dataの作成
valtest_transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda img: np.array(img)),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.float()),
    ])
valid_dataset = ImageDataset(train[45000:], transform=valtest_transform)
validloader = DataLoader(valid_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)

test_dataset = ImageDataset(test, transform=valtest_transform)
testloader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)

# def imshow(img):
#     """
#     functions to show an image
#     """
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.numpy().shape)
# # show images
# imshow(torchvision.utils.make_grid(images ))

def validate(net, validloader):
  """
  epoch毎に性能評価をするための関数
  """
  net.eval()
  correct = 0
  total = 0
  preds = torch.tensor([]).float().to(device)
  trues = torch.tensor([]).long().to(device)
  with torch.no_grad():
      for data in validloader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)

          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

          preds = torch.cat((preds, outputs))
          trues = torch.cat((trues, labels))
      val_loss = criterion(preds, trues)
      err_rate = 100 * (1 - correct / total)

  return val_loss, err_rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 学習用に必要なインスタンスを作成
net = ResNet50().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005,
                      momentum=0.9, weight_decay=0.0001)


scheduler = ReduceLROnPlateau(
                              optimizer,
                              mode='min',
                              factor=0.1,
                              patience=10,
                              verbose=True
                             )


# ロギング用のリスト
log = {'train_loss':[],
       'val_loss': [],
       'train_err_rate': [],
       'val_err_rate': []}

N_EPOCH = 40


# 学習を実行
for epoch in tqdm(range(N_EPOCH)):
    net.train()
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # epoch内でのlossを確認
        if i % 100 == 0:
            print(loss)

    else:
        # trainとvalに対する指標を計算
        train_loss, train_err_rate = validate(net, trainloader)
        val_loss, val_err_rate = validate(net, validloader)
        log['train_loss'].append(train_loss.item())
        log['val_loss'].append(val_loss.item())
        log['val_err_rate'].append(val_err_rate)
        log['train_err_rate'].append(train_err_rate)
        print(loss)
        print(f'train_err_rate:\t{train_err_rate:.1f}')
        print(f'val_err_rate:\t{val_err_rate:.1f}')
        scheduler.step(val_loss)
else:
    print('Finished Training')
