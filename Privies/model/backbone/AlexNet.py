import torch 
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # model info
        self.out_channel = 256
        self.in_channel = 3
        # self.feature_map_size = 
        self.eps = 1e-6
        # in 127,255
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2), # 59, 123
            nn.BatchNorm2d(96, eps=self.eps, momentum = 0.05),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(3,2), #(59-3)/2 +1 =29, 61 
 
            nn.Conv2d(96,256,5,1,groups=2), # 29-5 + 1 = 25 , 57
            nn.BatchNorm2d(256, eps = self.eps, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2), # (25-3)/2 +1 =12 , 28

            nn.Conv2d(256,384,3,1), # 12-3 +1 = 10, 26
            nn.BatchNorm2d(384, eps = self.eps, momentum=0.05),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,384,3,1), # 10-3 + 1 = 8, 24
            nn.BatchNorm2d(384, eps = self.eps, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1) # 10-3 +1 = 6, 22
        )
    
    def forward(self, x):
        x = self.feature(x)
        return x



if __name__ == "__main__":
    
    device = "cpu"
    test_image = torch.rand(1, 3, 255, 255).to(device)
    test_template = torch.rand(1,3,127,127).to(device)
    model = AlexNet()

    # model(test_template, test_image)
    print(model)