from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from backbone.HRNet import HRNetB
import numpy as np
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(planes * self.expansion, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes, momentum=norm_momentum)
        self.res_conv = None
        if inplanes != planes:
            self.res_conv = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        out += residual
        out = self.relu(out)

        return out





class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, norm_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, norm_momentum=norm_momentum),
                BasicBlock(w, w, norm_momentum=norm_momentum),
                BasicBlock(w, w, norm_momentum=norm_momentum),
                BasicBlock(w, w, norm_momentum=norm_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm3d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1,
                                      bias=False),
                            nn.BatchNorm3d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1,
                                  bias=False),
                        nn.BatchNorm3d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

    # def forward(self,x):
    #     x = checkpoint.checkpoint(self.forward1, x)
    #     return x


class HRNetB(nn.Module):
    def __init__(self, base=32, norm_momentum=0.1):
        super(HRNetB, self).__init__()
        c = base
        self.base_vec = torch.ones(1, c, 1, 1, 1)
        self.input_layer = BasicBlock(1, c)

        # Input (stem net)
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(base, c*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c*2, eps=1e-05, momentum=norm_momentum, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            BasicBlock(c*2, c*2))

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(c*2, c*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c*2, eps=1e-05, momentum=norm_momentum, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True))
        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv3d(c*2, c*8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(c*8, eps=1e-05, momentum=norm_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(c*2, c*2, downsample=downsample),
            Bottleneck(c*8, c*2),
            Bottleneck(c*8, c*2),
            Bottleneck(c*8, c*2),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(c*8, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(c, eps=1e-05, momentum=norm_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c*8, c * (2 ** 1), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c * (2 ** 1), eps=1e-05, momentum=norm_momentum, affine=True,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, norm_momentum=norm_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(c * (2 ** 1), c * (2 ** 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(c * (2 ** 2), eps=1e-05, momentum=norm_momentum, affine=True,
                               track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
               nn.Conv3d(c * (2 ** 2), c * (2 ** 3), kernel_size=3, stride=2, padding=1, bias=False),
               nn.BatchNorm3d(c * (2 ** 3), eps=1e-05, momentum=norm_momentum, affine=True,
                                 track_running_stats=True),
               nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
            StageModule(stage=3, output_branches=3, c=c, norm_momentum=norm_momentum),
        )

        # Final layer (final_layer)
        self.fuse_up_layer1 = nn.Sequential(nn.Conv3d(c * 7, c * 4, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm3d(c * 4, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose3d(c * 4, c * 2, 2, 2),
                                            nn.BatchNorm3d(c * 2, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True),
                                            BasicBlock(c * 2, c * 2)
                                            )
        self.fuse_up_layer2 = nn.Sequential(nn.Conv3d(c * 4, c * 2, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm3d(c * 2, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose3d(c * 2, c, 2, 2),

                                            nn.BatchNorm3d(c, eps=1e-05, momentum=norm_momentum, affine=True,
                                                           track_running_stats=True),
                                            nn.ReLU(inplace=True))
        self.fuse_layer = nn.Sequential(
            BasicBlock(c * 2, c),
            BasicBlock(c, c),
            nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )

        self.last_layer = nn.Sequential(
            nn.Conv3d(in_channels=c, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        self.pool = nn.AdaptiveAvgPool3d((25, 1, 1))

    def forward(self, x, logCube1):
        # x.requires_grad_()
        B, C, T, H, W = logCube1.shape
        x_skip1 = self.input_layer(x)       # 大小不变 扩展维度到32
        x_skip2 = self.conv_block1(x_skip1) # 下采样 2 扩展维度到64
        x = self.conv_block2(x_skip2)       # 下采样 2
        # x = x_skip2
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)

        # x = [
        #     self.transition3[0](x[0]),
        #     self.transition3[1](x[1]),
        #     self.transition3[2](x[2]),
        #     self.transition3[3](x[-1]),
        # ]  # New branch derives from the "upper" branch only
        x = self.stage4(x)

        # for i in range(len(self.basic)):
        #     x[i] = self.basic[i](x[i])
        #
        # for i in range(len(self.upper)):
        #     x[i] = self.upper[i](x[i])

        _, _, x0_t, x0_h, x0_w = x[0].shape
        x1 = F.interpolate(x[1], size=(x0_t, x0_h, x0_w), mode='trilinear')
        x2 = F.interpolate(x[2], size=(x0_t, x0_h, x0_w), mode='trilinear')

        x_torewarder = x[2]
        x_torewarder = self.pool(x_torewarder)
        x = torch.cat([x[0], x1, x2], 1)
        x = self.fuse_up_layer1(x)
        x = torch.cat([x_skip2, x], 1)
        x = self.fuse_up_layer2(x)

        x = torch.cat([x_skip1, x], 1)

        x_togenerater = self.fuse_layer(x)
        x = (F.cosine_similarity(x_togenerater, self.base_vec.to(x_togenerater.device), dim=1) + 1) / 2
        x = x[:, None]
        x_togenerater = F.interpolate(x_togenerater, size=(T, H, W), mode='trilinear')
        #x = self.last_layer(x_feat)
        x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=True)

        return x, x_togenerater

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def z_score_clip(data, clp_s=3.2):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))

backbone = HRNetB().to('cpu')
backbone.load_state_dict(torch.load('../model/F3fixmatch.pth', map_location='cpu'))
# backbone.eval()
print('F3fixmatch')
# model.eval()

all = z_score_clip(np.load('../data/F3/ORGSeis.npy'), 3.2)
print(all.shape)
val = all
all = all[None,None]
val = torch.from_numpy(val)[:,0:600,:]
print(val.shape)
val = F.interpolate(val[None,None], (400, 608, 960), mode='trilinear', align_corners=True)

val1 = val[:,:,:,0:304,:]
print(val1.shape)
# with autocast('cpu'):
val_output1, x_togenerater = backbone(val, val)
val_output1 = val_output1[0, 0, :]
val_output1 = val_output1.detach().numpy()

val1 = val[:,:,:,304:608,:]
print(val1.shape)
val_output2, x_togenerater = backbone(val, val)
val_output2 = val_output2[0, 0, :]
val_output2 = val_output2.detach().numpy()

val_output = np.concatenate((val_output1,val_output2),axis=-2)
np.save("../result/F3fixmatch.npy", val_output)
print(val_output.shape)

