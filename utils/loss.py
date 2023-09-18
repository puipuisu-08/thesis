import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEntropyLoss(nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
        b = b.sum(dim=2)
        b = -1.0 * b.mean(dim=1)
        return b.mean()


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_sz = int(source.size()[0])
        total_sum = []
        for i in range(batch_sz):
            s = source[i, :, :].permute(1, 0)
            t = target[i, :, :].permute(1, 0)
            n_samples = int(s.size()[0]) + int(t.size()[0])
            total = torch.cat([s, t], dim=0)
            total0 = total.unsqueeze(0).expand(
                int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(
                int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance = ((total0 - total1) ** 2).sum(2)
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i)
                            for i in range(kernel_num)]
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                        for bandwidth_temp in bandwidth_list]
            total_sum.append(sum(kernel_val))
        return sum(total_sum) / batch_sz

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            size = 128
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            
            with torch.no_grad():
                XX = torch.mean(kernels[:size, :size])
                YY = torch.mean(kernels[size:, size:])
                XY = torch.mean(kernels[:size, size:])
                YX = torch.mean(kernels[size:, :size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()
    
    def forward(self, source, target):
        d = source.size(1)
    
        # source covariance
        xm = torch.mean(source, 2, keepdim=True) - source
        xc = xm @ xm.permute(0, 2, 1)

        # target covariance
        xmt = torch.mean(target, 2, keepdim=True) - target
        xct = xmt @ xmt.permute(0, 2, 1)

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss
