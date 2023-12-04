import torch.nn as nn
import torch
from torch.autograd import Variable
from model.SiamRPN import SiamRPN
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    



def main():
    # torch.cuda.manual_seed(args.seed)
    model = SiamRPN()
    model = model.cuda()
    model = model.eval()

    criterion = nn.SoftMarginLoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5 * 1e-4)

    if not os.path.isdir("./cp/temp"):
        os.makedirs("./cp/temp")

    for epoch in range(100):
        cur_lr = adjust_learning_rate(optimizer, epoch) # 调整学习率

        """trainRPN"""
        loss = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        train_laoder = torch.utils.data.DataLoader(
            

        )




