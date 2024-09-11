from torch.cuda.amp import GradScaler, autocast
import os
import wandb
import torch
import numpy as np
from SemiDataSet import SemiDataset
from MyDataSet import seisDataset
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
if __name__ == '__main__':
    name = 'rewarderlog9' + datetime.now().strftime('%Y%m%d%H%M')
    wandb.init(project="Impedance", name=name)
    config = wandb.config

    '''
    文件路径
    '''
    config.seismic = 'data/denoise_seismic.npy'
    config.logCube = 'data/logCube_9.npy'

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
    config.lr = 0.001
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
    config.model_saved_step = 1000
    config.val_step = 1000
    config.loss_saved_step = 10
    config.batch_size = 2
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
    train_set = seisDataset(config, iters=config.batch_size * config.all_steps, seismic_road=config.seismic,
                            logCube_road=config.logCube,F3=False)
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size)

    '''
    初始化
    '''
    rewarder = Rewarder(400,400).to(config.device)
    generator = Generator().to(config.device)
    backbone = HRNetB().to(config.device)
    backbone.load_state_dict(torch.load('model/SYNHRNetlog9.pth'))
    backbone.eval()
    # model = FixMatch(config,backbone).to(config.device)
    # model.train()
    rewarder.train()
    generator.train()
    # optimizer = eval(f'torch.optim.{config.optim}')(model.parameters(), lr=config.lr)
    rewarder_optimizer = torch.optim.Adam(rewarder.parameters(), lr=config.sr_lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.sr_lr)
    loss_function = lambda_loss()
    loss_sum = []
    criterion = torch.nn.L1Loss()
    mse_max = 13000
    wandb.watch(rewarder, log="all")
    s_time = time.time()
    generator_loss_sum = []
    rewarder_loss_sum = []
    print('==============初始化完毕，开始进行训练=================')

    '''
    训练开始
    '''
    epoch = 0
    for idx, batch in enumerate(train_loader):
        # print('**************************************************************************')
        seismic, logCube = batch
        B, C, T, H, W = seismic.shape
        # seis = torch.cat((seismic, ul_seismic_w, ul_seismic_s))
        seis = seismic
        seis = resize_time(seis.to(config.device))
        logCube = logCube.to(config.device)
        # optimizer.zero_grad()
        # print(f'Initial seis dtype: {seis.dtype}')
        # print(f'Initial logCube dtype: {logCube.dtype}')
        with autocast(enabled=True):
            if idx < config.start_timing:
                logits_x_lb, togenerater_x_lb = backbone(seis, logCube)

                real_labels = []
                togenerater_feats = []
                predicted_labels = []
                for i in range(B):
                    log = logCube[i, :][0]
                    coordinates_x, coordinates_y = find_valid_coordinates(log)
                    real_labels.append(logCube[i, :, :, coordinates_x, coordinates_y])
                    togenerater_feats.append(togenerater_x_lb[i, :, :, coordinates_x, coordinates_y][None])
                    predicted_labels.append(logits_x_lb[i,:,:,coordinates_x, coordinates_y])

                # togenerater_x_lb = togenerater_x_lb.float()
                # print(f'togenerater_x_lb dtype before generator: {togenerater_x_lb.dtype}')
                togenerater_feats = torch.cat(togenerater_feats, dim=0)
                real_labels = torch.cat(real_labels, dim=0)
                predicted_labels = torch.cat(predicted_labels, dim=0)

                generated_labels = generator(togenerater_feats.detach())
                generated_labels = generated_labels.squeeze(-1)
                # print(generated_labels[0])
                cosine_similarity_score = (F.cosine_similarity(generated_labels.float(), real_labels.float(), dim=1) + 1) / 2
                cosine_similarity_score = cosine_similarity_score.view(generated_labels.size(0), 1)


                reward = rewarder(togenerater_feats.detach(), generated_labels.detach())
                # print("rewarder:",reward[0],reward[1])
                generator_loss = criterion(cosine_similarity_score, torch.ones_like(cosine_similarity_score))
                rewarder_loss = criterion(reward, cosine_similarity_score)

                generator_loss_sum.append(generator_loss.item())
                rewarder_loss_sum.append(rewarder_loss.item())

                generator_optimizer.zero_grad()
                rewarder_optimizer.zero_grad()

                # generator_loss.backward(retain_graph=True)
                # rewarder_loss.backward(retain_graph=True)

                scaler.scale(generator_loss).backward(retain_graph=True)
                scaler.step(generator_optimizer)
                scaler.update()

                scaler.scale(rewarder_loss).backward(retain_graph=True)
                scaler.step(rewarder_optimizer)
                scaler.update()

                # generator_optimizer.step()
                # rewarder_optimizer.step()

                # print("generator_loss", generator_loss.item())
                # print("rewarder_loss", rewarder_loss.item())
                if idx % config.loss_saved_step == 0 and idx != 0:

                    print("Iteration:%d, rewarder_loss:%.6f, generator_loss:%.6f" % (idx, np.mean(rewarder_loss_sum), np.mean(generator_loss_sum)))

                    wandb.log({
                    "generator_loss": np.mean(generator_loss_sum),
                    "rewarder_loss": np.mean(rewarder_loss_sum)
                    })
                    generator_loss_sum = []
                    rewarder_loss_sum = []

                if (idx % config.val_step == 0 and idx != 0 ) or (idx == config.all_steps - 1):
                    rewarder_road = os.path.join('/root/autodl-tmp/rewarder', str(idx) + 'model.pth')
                    torch.save(rewarder.state_dict(), rewarder_road)
                    wandb.save(rewarder_road)

                    generator_road = os.path.join('/root/autodl-tmp/generator', str(idx) + 'model.pth')
                    torch.save(generator.state_dict(), generator_road)
                    wandb.save(generator_road)
            else:
                rewarder.eval()
                # output = model(seis,logCube,epoch)


        epoch = epoch + 1