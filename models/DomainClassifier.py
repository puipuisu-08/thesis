import torch.nn as nn
from layers.ReverseLayerF import ReverseLayerF

class DANN_Default(nn.Module):
    def __init__(self, args):
        super().__init__()

        patch_num = int((args.seq_len - args.patch_len)/args.stride + 1)
        if args.padding_patch == 'end': # can be modified to general case
            patch_num += 1

        self.head_nf = args.d_model * patch_num

        self.fc1 = nn.Linear(self.head_nf, 64)
        self.bn1 = nn.BatchNorm1d(args.enc_in)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x, alpha):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        x = self.fc1(reverse_feature)
        x = self.bn1(x)
        x = self.relu1(x)
        domain_output = self.fc2(x)
        return domain_output
    
class DANN_AdaTime(nn.Module):
    def __init__(self, args):
        super().__init__()

        patch_num = int((args.context_window - args.patch_len)/args.stride + 1)
        if args.padding_patch == 'end': # can be modified to general case
            patch_num += 1

        self.head_nf = args.d_model * patch_num

        self.fc1 = nn.Linear(self.head_nf, 128)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, alpha):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        x = self.fc1(reverse_feature)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        domain_output = self.fc3(x)
        return domain_output

