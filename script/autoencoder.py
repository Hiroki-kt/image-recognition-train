import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml

# mnistのデータをダウンロード
mnist = fetch_openml('mnist_784', version=1, data_home='data/src/download/')
X, y = mnist["data"], mnist["target"]
X_0 = X[np.int32(y) == 0]  # targetデータが0の場合のみを抜出
X_0 = (2 * X_0) / 255.0 - 1.0  # max 1.0 min -1.0 に変換

X_0 = X_0.reshape([X_0.shape[0], 1, 28, 28])
X_0 = torch.tensor(X_0, dtype=torch.float32)
X_0_train, X_0_test = X_0[:6000, :, :, :], X_0[6000:, :, :, :]
train_loader = DataLoader(X_0_train, batch_size=50)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4,
                               kernel_size=3, padding=1)
        # Decoder Layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16,
                                          kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                          kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # コメントに28×28のモノクロ画像をi枚を入力した時の次元を示す
        # encode#                          #in  [i, 1, 28, 28]
        x = self.relu(self.conv1(x))  # out [i, 16, 28, 28]
        x = self.pool(x)  # out [i, 16, 14, 14]
        x = self.relu(self.conv2(x))  # out [i, 4, 14, 14]
        x = self.pool(x)  # out [i ,4, 7, 7]
        #decode#
        x = self.relu(self.t_conv1(x))  # out [i, 16, 14, 14]
        x = self.sigmoid(self.t_conv2(x))  # out [i, 1, 28, 28]
        return x


def train_net(n_epochs, train_loader, net, optimizer_cls=optim.Adam,
              loss_fn=nn.MSELoss(), device="cpu"):
    """
    n_epochs…訓練の実施回数
    net …ネットワーク
    device …"cpu" or "cuda:0"
    """
    losses = []  # loss_functionの遷移を記録
    optimizer = optimizer_cls(net.parameters(), lr=0.001)
    net.to(device)

    for epoch in range(n_epochs):
        running_loss = 0.0
        net.train()  # ネットワークをtrainingモード

        for i, XX in enumerate(train_loader):
            XX.to(device)
            optimizer.zero_grad()
            XX_pred = net(XX)  # ネットワークで予測
            loss = loss_fn(XX, XX_pred)  # 予測データと元のデータの予測
            loss.backward()
            optimizer.step()  # 勾配の更新
            running_loss += loss.item()

        losses.append(running_loss / i)
        print("epoch", epoch, ": ", running_loss / i)

    return losses


net = ConvAutoencoder()
losses = train_net(n_epochs=30,
                   train_loader=train_loader,
                   net=net)

img_num = 4
pred = net(X_0_test[img_num:(img_num + 1)])
pred = pred.detach().numpy()
pred = pred[0, 0, :, :]

origin = X_0_test[img_num:(img_num + 1)].numpy()
origin = origin[0, 0, :, :]

plt.subplot(211)
plt.imshow(origin, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.text(x=3, y=2, s="original image", c="red")

plt.subplot(212)
plt.imshow(pred, cmap="gray")
plt.text(x=3, y=2, s="output image", c="red")
plt.xticks([])
plt.yticks([])
plt.savefig("./data/0_auto_encoder")
plt.show()
