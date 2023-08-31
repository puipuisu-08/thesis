import torch
import torch.nn as nn

class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()
    
    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 2, keepdim=True) - source
        xc = xm.permute(0, 2, 1) @ xm

        # target covariance
        xmt = torch.mean(target, 2, keepdim=True) - target
        xct = xmt.permute(0, 2, 1) @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss