from torch.cuda.amp import GradScaler, autocast
import os
import wandb
import torch
import numpy as np
from SemiDataSet import SemiDataset
from torch.utils.data import DataLoader
from utils.loss_class import lambda_loss
from sklearn.metrics import mean_squared_error
import time
from torch.nn import functional as F
import random
from utils.dataAug import z_score_clip
from matplotlib import pyplot as plt
from datetime import datetime

from backbone.HRNet import HRNetB
from algorithms.semireward import Generator,Rewarder

from algorithms.fixmatch import FixMatch
scaler = GradScaler(enabled=True)
def resize_time(seis):
    if random.random() < 0.2:
        return seis
    if random.randint(0, 1):
        time_scale = random.uniform(1, 2)
    else:
        time_scale = random.uniform(0.5, 1)
    resize_time = int(round(T * time_scale))
    resize_time = resize_time + 16 - (resize_time % 16)
    seis = F.interpolate(seis, (resize_time, H, W), mode='trilinear', align_corners=True)
    return seis

def find_valid_coordinates(data):
    for a in range(data.size(1)):
        for b in range(data.size(2)):
            # 计算当前 (a, b) 坐标处的平均值
            avg_value = torch.mean(data[:, a, b])
            # 检查平均值是否不为 -999
            if avg_value != -999:
                return a, b  # 返回第一个满足条件的坐标 (a, b)
    return None  # 如果找不到满足条件的坐标，则返回 None

def logCubenorm(logCube):
    logCube = np.where(logCube != -999, np.clip(logCube, a_min=3500, a_max=6000),logCube)
    MAX = np.max(logCube)
    MIN = np.min(np.where(logCube == -999, 9999999, logCube))
    logVec = np.where(logCube != -999, (logCube - MIN) / (MAX - MIN), logCube)
    print(MAX,MIN,np.min(logCube))
    return logVec,MAX,MIN

if __name__ == '__main__':
    name = 'fixmatchhrnetlog3'
    wandb.init(project="Impedance", name=name)
    config = wandb.config

    '''
    文件路径
    '''
    config.seismic = 'data/F3/ORGSeis.npy'
    config.logCube = 'data/F3/AI.npy'

    '''
    数据集配置
    '''
    config.normal_clip = 3.2
    config.crop_lim = (0.5, 2)
    config.train_cube_size = 48
    config.m_cover = 5

    '''
    模型参数
    '''
    config.base_width = 32
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    优化器
    '''
    config.optim = 'AdamW'
    config.lr = 0.0005
    config.sr_lr = 0.0005

    '''
    损失函数
    '''
    config.loss = 'l1'

    '''
    实验参数
    '''
    config.all_steps = 50000
    config.start_timing = 20000
    config.model_saved_step = 500
    config.val_step = 50
    config.loss_saved_step = 10
    config.batch_size = 1
    config.seed = 3
    config.restore = None
    random.seed(config.seed)
    np.random.seed(config.seed)

    '''
    读取地震数据、阻抗数据
    '''
    all = z_score_clip(np.load(config.seismic), config.normal_clip)
    print(all.shape)
    all = torch.from_numpy(all)
    all = F.interpolate(all[None, None], (400, 672, 960), mode='trilinear', align_corners=True)
    a1 = np.zeros((400, 672, 960),dtype=float)
    aaa = 0

    imp = np.load(config.logCube)

    imp1 = np.where(imp != -999, np.clip(imp, a_min=3500, a_max=6000),imp)
    MAX = np.max(imp1)
    MIN = np.min(np.where(imp1 == -999, 9999999, imp1))
    logVec = np.where(imp1 != -999, (imp1 - MIN) / (MAX - MIN), imp1)
    v1 = logVec[:, 143, 87]
    v2 = logVec[:, 144, 86]
    v3 = logVec[:, 144, 87]
    v4 = logVec[:, 144, 88]
    v5 = logVec[:, 145, 87]


    '''
    加载数据
    '''
    train_set = SemiDataset(config, iters=config.batch_size * config.all_steps, seismic_road=config.seismic,
                            logCube_road=config.logCube, F3=True)
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size)

    '''
    初始化
    '''
    rewarder = Rewarder(400, 400).to(config.device)
    rewarder.load_state_dict(torch.load('rewarder_log3/5000model.pth'))
    rewarder.eval()

    backbone = HRNetB().to(config.device)
    backbone.load_state_dict(torch.load('model/F3HRNetBatch34000model.pth'))
    backbone.train()
    optimizer = eval(f'torch.optim.{config.optim}')(backbone.parameters(), lr=config.lr)

    loss_function = lambda_loss()
    criterion = torch.nn.L1Loss()
    loss_sum = []
    val_loss_sum = []
    val_min = 0.005
    val_min_list = []
    mse_max = 1e9
    wandb.watch(backbone, log="all")
    s_time = time.time()
    print('==============初始化完毕，开始进行训练=================')

    for idx, batch in enumerate(train_loader):
        seismic_w, seismic_s, logCube = batch
        B, C, T, H, W = logCube.shape
        seis = torch.cat((seismic_w, seismic_s))
        seis = resize_time(seis.to(config.device))
        logCube = logCube.to(config.device)
        optimizer.zero_grad()

        with autocast(enabled=True):
            output, x_feat = backbone(seis, logCube)
            output_w = output[:B]
            output_s = output[B:]
            x_feat_w = x_feat[:B]
            x_feat_s = x_feat[B:]
            sup_loss = loss_function(output_s, logCube, config.loss)
            toreward_feats_s = []

            for i in range(B):
                log = logCube[i, :][0]
                coordinates_x, coordinates_y = find_valid_coordinates(log)
                toreward_feats_s.append(x_feat_s[i, :, :, coordinates_x, coordinates_y][None])
                # toreward_feats_s.append(x_feat_s[i, :, :, coordinates_x, coordinates_y][None])
            toreward_feats_s = torch.cat(toreward_feats_s, dim=0)

            coordinates = []
            fake_logCube = torch.ones_like(logCube) - 1000.
            empty_flag = False
            for x in range(output.size(-2)):
                for y in range(output.size(-1)):
                    a = output_w[:, :, :, x, y].squeeze(1)
                    reward = rewarder(toreward_feats_s.detach(), a.detach())
                    for i in range(B):
                        # print("reward:",reward[i].item())
                        if reward[i].item() > 0.998:
                            fake_logCube[i, :, :, x, y] = output_w[i, :, :, x, y]

            indices = torch.where(fake_logCube != -999.)
            if indices[0].numel() == 0:
                print('空')
                continue
            unsup_loss = loss_function(output_s, fake_logCube, config.loss)
            train_loss = sup_loss + 0.5*unsup_loss
            train_loss = train_loss.mean()
            loss = train_loss
            loss_sum.append(train_loss.item())
            print('loss:', loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if idx % config.loss_saved_step == 0 and idx != 0:
            e_time = time.time()
            int_time = e_time - s_time
            print("Iteration:%d, Loss:%.6f,time_taken:%.2f" % (idx, np.mean(loss_sum), int_time))
            s_time = time.time()
            wandb.log({
                "train_loss": np.mean(loss_sum)
            })
            loss_sum = []

        if (idx % config.val_step == 0 and idx != 0 ) or (idx == config.all_steps - 1):
            backbone.eval()
            with torch.no_grad():
                with autocast(enabled=True):
                    for i in range(1, 15):
                        for j in range(1, 21):
                            val = all[:, :, :, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j].to(config.device)
                            val_output, val_feats = backbone(val, val)
                            val_output = val_output.cpu().numpy()[0, 0]
                            a1[:, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j] = val_output

            val_target = a1[:, 240, :]
            plt.imshow(val_target, cmap='jet')
            target_image = wandb.Image(plt)
            wandb.log({"target_image": target_image})

            if idx > 40:
                a1 = torch.from_numpy(a1)
                a1 = F.interpolate(a1[None, None], (400, 651, 951), mode='trilinear', align_corners=True)
                a1 = a1.cpu().numpy()[0, 0]
                p1 = a1[:, 143, 87]
                p2 = a1[:, 144, 86]
                p3 = a1[:, 144, 87]
                p4 = a1[:, 144, 88]
                P5 = a1[:, 145, 87]
                mse = (np.sum((p1 - v1)**2) / 400 + np.sum((p2 - v2)**2) / 400 + np.sum((p3 - v3)**2) / 400 + np.sum((p4 - v4)**2) / 400)/4
                mae = (np.sum(np.abs(p1 - v1)) / 400 + np.sum(np.abs(p2 - v2)) / 400 + np.sum(np.abs(p3 - v3)) / 400+ np.sum(np.abs(p4 - v4)) / 400)/4

                wandb.log({
                    "mse": mse,
                    "mae": mae
                })
                print('mse:',mse)
                print('mae:',mae)
                # if mse_max > mse:
                #     mse_max = mse
                #     road = os.path.join('/root/autodl-tmp', str(mse) + 'model.pth')
                #     torch.save(backbone.state_dict(), road)
                #     wandb.save(road)
                road = os.path.join('/root/autodl-tmp', str(idx) + str(mae) + 'model.pth')
                torch.save(backbone.state_dict(), road)
                wandb.save(road)
            a1 = np.zeros((400, 672, 960), dtype=float)
            # np.save(str(aaa)+'npy',a1)
            # if idx >30000:
            #     aaa += 1
            #     np.save(r'/root/autodl-tmp/'+str(aaa),a1)
            backbone.train()
    print(val_min_list)
    wandb.finish()

