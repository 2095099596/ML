import torch
import torch.nn as nn
import torch.optim as optim
import torch
# torch.autograd.set_detect_anomaly(True)
import pandas as pd
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
from einops.layers.torch import Rearrange

def Two_to_Tress(x):
    df = pd.DataFrame(x)
    # 将 DataFrame 转换为 NumPy 数组
    data_np = df.to_numpy()
    reshaped_data = data_np.reshape(40,48, 18, 6)
    # reshaped_data = reshaped_data.reshape(41, 48, 18, 6)
    return  reshaped_data
'''
读取数据，并将数据变为三维数据（实验数据包括两个特征，降水数据和蒸散发数据）
'''



'''
将数据存放在Datalador中
'''
PR = pd.read_excel('data/PR.xlsx')
EV = pd.read_excel('data/EV.xlsx')
PR2 = PR.values
PR3 = PR2[59:,31:]
EV2 = EV.values
EV3 = EV2[59:,31:]
# print(PR3.shape)
PR4 = Two_to_Tress(PR3)
EV4 = Two_to_Tress(EV3)
Data = np.stack((PR4, EV4), axis=1)
print(Data.shape)
Data = Data.astype(np.float32)
Data = torch.tensor(Data, dtype=torch.float32)  # 你可以指定数据类型

# print(Data.dtype)
'''

'''
X = Data[:35,:,:36, :, :]
y = Data[:35,0,36:, 9, 3]
dataset = TensorDataset(X,y)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

X_test = Data[35:,:,:36, :, :]
y_test = Data[35:,0,36:,9, 3]




class ConvLSTMModel(nn.Module):
    def __init__(self, future_days):
        super(ConvLSTMModel, self).__init__()
        # 3D卷积层
        self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))
        )

        # LSTM层参数
        self.hidden_size = 50
        self.num_layers = 1

        # LSTM层，输入特征维度为25（5*5），因为我们在池化步骤后有8个时间步
        self.lstm = nn.LSTM(input_size=25, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # 全连接层，用于时间序列预测的输出，预测未来future_days天的某个特征
        self.fc = nn.Linear(self.hidden_size, future_days)
        self.norm =  nn.LayerNorm([12, 6, 2])

    def forward(self, x):
        # 应用3D卷积层
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        #x = self.norm(x)
        print(x)
        # 调整形状以匹配LSTM输入要求：(batch_size, seq_len, input_size)
        x = x.view(x.size(0), 12, -1)  # 假设batch_size=1，这里调整形状为(1, 8, 25)
        print(x.shape)

        # 通过LSTM层
        x, _ = self.lstm(x)
        #print(x.shape)
        # 只取序列中的最后一个时间步的输出
        x = x[:, -1, :]
        #print(x.shape)
        # 应用全连接层得到最终的预测结果，预测未来10天的数据
        x = self.fc(x)
        #print(x.shape)
        return x



class ConvLSTMModel(nn.Module):
    def __init__(self, future_days):
        super(ConvLSTMModel, self).__init__()
        # 3D卷积层
        self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
        )

        # LSTM层参数
        self.hidden_size = 50
        self.num_layers = 1

        # LSTM层，输入特征维度为25（5*5），因为我们在池化步骤后有8个时间步
        self.lstm = nn.LSTM(input_size=12, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # 全连接层，用于时间序列预测的输出，预测未来future_days天的某个特征
        self.fc = nn.Linear(self.hidden_size, future_days)
        #self.norm =  nn.LayerNorm([36, 6, 2])

    def forward(self, x):
        # 应用3D卷积层
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        #x = self.norm(x)
        # print(x)
        # 调整形状以匹配LSTM输入要求：(batch_size, seq_len, input_size)
        x = x.view(x.size(0), 36, -1)  # 假设batch_size=1，这里调整形状为(1, 8, 25)
        #print(x.shape)

        # 通过LSTM层
        x, _ = self.lstm(x)
        #print(x.shape)
        # 只取序列中的最后一个时间步的输出
        x = x[:, -1, :]
        #print(x.shape)
        # 应用全连接层得到最终的预测结果，预测未来10天的数据
        x = self.fc(x)
        #print(x.shape)
        return x







model = ConvLSTMModel(future_days=12)  # 假设已定义该模型
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# model = ViT()
print(model)
# Loss Function and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)





# 将模型设置为训练模式
model.train()
num_epochs = 40
Loss=[]
# 迭代数据
for epoch in range(num_epochs):  # num_epochs是你希望训练的总轮数
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # 清空之前的梯度

        # 前向传播
        #print(inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # 计算损失

        # 后向传播和优化
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        Loss.append(loss.item())

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.plot(Loss, marker='o', linestyle='-', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

outputs = outputs.detach()
outputs = outputs.numpy()  # 转换为 NumPy 数组


from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

outputs = outputs.flatten()
y_test = y_test.flatten()
r2 = r2_score(outputs, y_test)

# 计算MSE
mse = mean_squared_error(outputs, y_test)

# 计算RMSE
rmse = np.sqrt(mse)

print(r2,rmse,mse)