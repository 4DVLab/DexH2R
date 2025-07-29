import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        # self.iden = torch.eye(3, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu").view(1, 9).repeat(batchsize, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
    
        x = F.relu_(self.bn4(self.fc1(x)))
        x = F.relu_(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, dtype=torch.float32, device="cuda" ).view(1, 9).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self , channel=3):

        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, x):

        trans = self.stn(x.transpose(2,1))     # let the pcd from [B，N，D] to [B,D,N]
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.max(dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x # , trans, trans_feat






# class STN3dV2(nn.Module):
#     def __init__(self, channel):
#         super(STN3dV2, self).__init__()
#         self.conv1 = torch.nn.Conv1d(channel, 256, 1)
#         self.conv2 = torch.nn.Conv1d(256, 512, 1)
#         self.conv3 = torch.nn.Conv1d(512, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)

#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#         # self.iden = torch.eye(3, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu").view(1, 9).repeat(batchsize, 1)

#     def forward(self, x):
#         batch_size = x.shape[0]

#         x = F.relu_(self.bn1(self.conv1(x)))
#         x = F.relu_(self.bn2(self.conv2(x)))
#         x = F.relu_(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
    
#         x = F.relu_(self.bn4(self.fc1(x)))
#         x = F.relu_(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = torch.eye(3, dtype=torch.float32, device="cuda").view(1, 9).repeat(batch_size, 1)
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x


# class PointNetEncoderV2(nn.Module):
#     def __init__(self , channel=3):

#         super(PointNetEncoderV2, self).__init__()
#         self.stn = STN3dV2(channel)
#         self.conv1 = torch.nn.Conv1d(channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)


#     def forward(self, x):

#         trans = self.stn(x.transpose(2,1))     # let the pcd from [B，N，D] to [B,D,N]
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
#         x = F.relu_(self.bn1(self.conv1(x)))
#         x = F.relu_(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = x.max(dim=2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         return x # , trans, trans_feat
















if __name__ == '__main__':
    model = PointNetEncoder()
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            x = torch.randn(1, 4096, 3).to("cuda")
            x = model(x)

    print(torch.cuda.memory_summary(device=torch.device("cuda"), abbreviated=False))