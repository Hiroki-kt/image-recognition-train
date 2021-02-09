import os
import urllib
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
from matplotlib import pyplot as plt


def validate(net, validloader):
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
            _, predicted = torch.max(outputs.data, 1)  # get max?
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            preds = torch.cat((preds, outputs))
            trues = torch.cat((trues, labels))
        val_loss = criterion(preds, trues)
        err_rate = 100 * (1 - correct / total)

    return val_loss, err_rate


if __name__ == '__main__':
    BATCH = 64
    DATA_DIR = './data/cifar-10-tensor/'

    model = models.resnet18()
    # model = models.resnet50()
    # model = models.resnet101()

    # load dataset
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)

    # make data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH, shuffle=False, num_workers=2)

    # check gpu environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # set loss and optimize function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005,
                          momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True
    )

    N_EPOCH = 40

    # logging
    log = {'train_loss': [],
           'val_loss': [],
           'train_err_rate': [],
           'val_err_rate': []}

    # traing
    for epoch in tqdm(range(N_EPOCH)):
        # print("epoch number", epoch)
        model.train()
        for i, data in tqdm(enumerate(train_loader, 0)):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # epoch内でのlossを確認
            if i % 100 == 0:
                print(loss)
        else:
            # trainとvalに対する指標を計算
            train_loss, train_err_rate = validate(model, train_loader)
            val_loss, val_err_rate = validate(model, test_loader)
            log['train_loss'].append(train_loss.item())
            log['val_loss'].append(val_loss.item())
            log['val_err_rate'].append(val_err_rate)
            log['train_err_rate'].append(train_err_rate)
            print(loss)
            print(f'train_err_rate:\t{train_err_rate:.1f}')
            print(f'val_err_rate:\t{val_err_rate:.1f}')
            # scheduler.step(val_loss)
    else:
        print('Finished Training')


"""
# Download ImageNet labels
# if you are mac user
# curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt > imagenet_classes.txt
# if you are linux user
# wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Download an example image from the pytorch website
url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# Read the categories
with open("/data/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
"""
