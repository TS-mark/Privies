import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.AlexNet import AlexNet


class SiamRPN(nn.Module):
    def __init__(self):
        super(SiamRPN, self).__init__()
        self.width = 256
        self.height = 256
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.backbone = AlexNet()
        
        self.regress_adjust = nn.Conv2d(4*5,4*5,1)
        # self.mid()
        self._initialize_weights()

    # def mid(self):
        self.conv_cls1 = nn.Conv2d(self.backbone.out_channel, self.backbone.out_channel *2*5, 
                                    kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(self.backbone.out_channel, self.backbone.out_channel * 4 * 5, 
                                    kernel_size=3, stride=1, padding=0)
        self.conv_cls2 = nn.Conv2d(self.backbone.out_channel, self.backbone.out_channel,
                                    kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(self.backbone.out_channel, self.backbone.out_channel,
                                    kernel_size=3, stride=1, padding=0)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def xcorr(self, z,x, channels):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            # F.conv2d(z, x) -->输入张量z，四维(bchw)，x：卷积核张量，四维(output_channels,input_channels,kernel_height,kernel_width)
            out.append(F.conv2d(x[i, :, :, :].unsqueeze(0),
                                z[i, :, :, :].unsqueeze(0).view(channels, self.backbone.out_channel, kernel_size, kernel_size)
                                ))
            
        return torch.cat(out, dim = 0)
    
    def forward(self, template, detection):
        template_feature = self.backbone(template)
        detection_feature = self.backbone(detection)

        kernel_score = self.conv_cls1(template_feature)
        kernel_regression = self.conv_r1(template_feature)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        pred_score = self.xcorr(kernel_score, conv_score, 10)
        pred_regression = self.regress_adjust(self.xcorr(kernel_regression, conv_regression, 20))
        return pred_score, pred_regression
    

if __name__ == "__main__":
    device = "cpu"
    test_image = torch.rand(1, 3, 255, 255).to(device)
    test_template = torch.rand(1,3,127,127).to(device)
    model = SiamRPN()
    # model(test_template, test_image)
    print(model)