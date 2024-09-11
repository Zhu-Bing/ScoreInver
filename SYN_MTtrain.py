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
from algorithms.semireward import Rewarder
from algorithms.MT import MeanTeacher
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


if __name__ == '__main__':
    name = 'MThrnetlog9w/oscorer'
    wandb.init(project="Impedance", name=name)
    config = wandb.config

    '''
    文件路径
    '''
    config.seismic = 'data/denoise_seismic.npy'
    config.logCube = 'data/logCube_9.npy'
    config.model_path = 'model/SYNHRNetlog9.pth'
    config.rewarder_path = 'rewarder/log9_3000model.pth'
    config.tau = 0

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
    config.batch_size = 2
    config.seed = 3
    config.restore = None
    config.momentum = 0.995
    random.seed(config.seed)
    np.random.seed(config.seed)


    '''
    读取地震数据、阻抗数据
    '''
    all = z_score_clip(np.load(config.seismic), config.normal_clip)
    print(all.shape)
    all = torch.from_numpy(all)
    all = F.interpolate(all[None, None], (400, 528, 528), mode='trilinear', align_corners=True)

    imp = (np.load('data/imp.npy').astype(np.float32))
    val_target = imp[:, 240, :]
    plt.imshow(val_target, cmap='jet')
    target_image = wandb.Image(plt)
    wandb.log({"imp": target_image})

    a1 = np.zeros((400, 528, 528),dtype=float)
    aaa = 0

    '''
    加载数据
    '''
    # semi_train_set = SemiDataset(config, iters=config.batch_size * config.all_steps, seismic_road=config.seismic,
    #                         logCube_road=config.logCube,F3=False)
    train_set = SemiDataset(config, iters=config.batch_size * config.all_steps, seismic_road=config.seismic,
                            logCube_road=config.logCube,F3=False)
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size)

    '''
    初始化
    '''
    rewarder = Rewarder(400,400).to(config.device)
    rewarder.load_state_dict(torch.load(config.rewarder_path))
    rewarder.eval()

    backbone1 = HRNetB().to(config.device)
    backbone2 = HRNetB().to(config.device)
    model = MeanTeacher(backbone1,backbone2, config).to(config.device)
    model.train()
    optimizer = eval(f'torch.optim.{config.optim}')(model.parameters(), lr=config.lr)

    loss_function = lambda_loss()
    loss_sum = []
    criterion = torch.nn.L1Loss()
    mse_max = 13000
    wandb.watch(model, log="all")
    s_time = time.time()
    print('==============初始化完毕，开始进行训练=================')

    '''
    训练开始
    '''
    for idx, batch in enumerate(train_loader):
        seismic_w, seismic_s, logCube = batch
        B, C, T, H, W = logCube.shape
        seismic_w = seismic_w.to(config.device)
        seismic_s = resize_time(seismic_s.to(config.device))
        logCube = logCube.to(config.device)
        optimizer.zero_grad()

        with autocast(enabled=True):
            output_s, x_feat_s, output_w, x_feat_w = model(idx, seismic_w, seismic_s, logCube)

            sup_loss = loss_function(output_s, logCube, config.loss)
            # print(f'sup_loss.requires_grad: {sup_loss.requires_grad}')
            toreward_feats_s = []

            for i in range(B):
                log = logCube[i, :][0]
                coordinates_x, coordinates_y = find_valid_coordinates(log)
                toreward_feats_s.append(x_feat_s[i, :, :, coordinates_x, coordinates_y][None])
            toreward_feats_s = torch.cat(toreward_feats_s, dim=0)

            coordinates = []
            # fake_logCube = new_logCube = np.zeros((tline, rH, rW)).astype(np.float32) - 999.
            fake_logCube = torch.ones_like(logCube)-1000.
            for x in range(output_s.size(-2)):
                for y in range(output_s.size(-1)):
                    a = output_w[:,:,:,x,y].squeeze(1)
                    reward = rewarder(toreward_feats_s.detach(), a.detach())
                    for i in range(B):
                        # print("reward:", reward[i].item())
                        if reward[i].item() > config.tau:
                            fake_logCube[i,:,:,x,y] = output_w[i,:,:,x,y]
                            # print(reward[i])
                            # coordinates.append((x,y))
                    # print('************************************************************')
            # coordinates
            unsup_loss = loss_function(output_s, fake_logCube, config.loss)
            # mask = torch.where(fake_logCube == -999., 0., 1.)
            train_loss = sup_loss + unsup_loss
            # train_loss = sup_loss
            train_loss = train_loss.mean()
            # print(f'train_loss.requires_grad: {train_loss.requires_grad}')
            # loss = train_loss
            loss_sum.append(train_loss.item())

            print('loss:', train_loss.item())

        scaler.scale(train_loss).backward()
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
            model.eval()
            with torch.no_grad():
                with autocast(enabled=True):
                    for i in range(1, 12):
                        for j in range(1, 12):
                            val = all[:, :, :, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j].to(config.device)
                            output_s, x_feat_s, output_w, x_feat_w = model(idx, val, val, logCube)
                            output_w = output_w.cpu().numpy()[0, 0]
                            a1[:, 48 * (i - 1):48 * i, 48 * (j - 1):48 * j] = output_w

            val_target = a1[:, 240, :]
            plt.imshow(val_target, cmap='jet')
            target_image = wandb.Image(plt)
            wandb.log({"target_image": target_image})

            if idx > 40:
                a1 = torch.from_numpy(a1)
                a1 = F.interpolate(a1[None, None], (400, 502, 501), mode='trilinear', align_corners=True)
                a1 = a1.cpu().numpy()[0, 0]
                mse = mean_squared_error(imp.flatten(),a1.flatten())
                mae = np.mean(np.abs(imp - a1))
                wandb.log({
                    "mse": mse,
                    "mae": mae
                })
                print('mse:',mse)
                print('mae:',mae)
                if mse_max > mse:
                    mse_max = mse
                    road = os.path.join('/root/autodl-tmp', str(mse) + 'model.pth')
                    torch.save(model.state_dict(), road)
                    wandb.save(road)
            a1 = np.zeros((400, 528, 528), dtype=float)
            # np.save(str(aaa)+'npy',a1)
            # if idx >30000:
            #     aaa += 1
            #     np.save(r'/root/autodl-tmp/'+str(aaa),a1)
            model.train()