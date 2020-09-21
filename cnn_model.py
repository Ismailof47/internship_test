import torch
import torch.nn as nn
import torch.nn.functional as F 


class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,96,kernel_size=7,stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.fc1 = nn.Linear(13824,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,2)
        self.apply(weights_init)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        #print out.size()
        out = (F.relu(self.fc1(out)))
        out = (F.relu(self.fc2(out)))
        out = self.fc3(out)

        return out




def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)



if __name__ == '__main__':

	pass
