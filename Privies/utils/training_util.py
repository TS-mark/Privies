def adjust_learning_rate(optimizer, epoch, ori_learning=1e-3, steps = [-1,1,40,60,70], scales = [.1,10,.1,.1,.1]):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = ori_learning
    
    for i in range(len(steps)):
        
        scale = scales[i] if i < len(scales) else 1

        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr