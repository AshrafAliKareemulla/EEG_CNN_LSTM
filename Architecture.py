import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(5, 64, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout1 = nn.Dropout2d(0.3)
        
        # Conv Block 2
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout2 = nn.Dropout2d(0.2)
        
        # Conv Block 3
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Conv Block 4
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout4 = nn.Dropout2d(0.3)
        
        # FFNN
        self.fc1 = nn.Linear(3*16*512, 512)
        self.dropout5 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 64)
        self.dropout7 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(64, 3)
        

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout1(self.maxpool1(x))
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout2(self.maxpool2(x))
        
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.maxpool3(x))
        
        x = F.relu(self.conv7(x))
        x = self.dropout4(self.maxpool4(x))
        
        x = x.view(-1, 3*16*512)
        
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.dropout7(F.relu(self.fc3(x)))
        
        x = self.fc4(x)
        
        return torch.softmax(x, dim = 1)











class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(62, 64, num_layers = 1, batch_first = True, dropout = 0.25)
        self.lstm2 = nn.LSTM(64, 128, num_layers = 1, batch_first = True, dropout = 0.3)
        self.lstm3 = nn.LSTM(128, 128, num_layers = 1, batch_first = True, dropout = 0.35)
        self.lstm4 = nn.LSTM(128, 256, num_layers = 1, batch_first = True, dropout = 0.4)
        
        self.fc1 = nn.Linear(256*1325, 512)
        self.dropout1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.35)
        
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        
        x = x.contiguous().view(-1, 256 * 1325)
        
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return torch.softmax(x, dim = 1)
    













class Hybrid_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(5, 64, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(128)
        
        
        # Conv Block 2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(5,5), padding=(2,2), stride=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout2 = nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm2d(256)
        
        # Conv Block 3
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(512)
        
        


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn1(self.dropout1(self.maxpool1(x)))
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.bn2(self.dropout2(self.maxpool2(x)))
        
        x = self.bn3(F.relu(self.conv6(x)))
        
        return x