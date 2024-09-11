from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from utils.dataAug import RandomHorizontalFlipCoord, RandomVerticalFlipCoord, RandomNoise, RandomRotateCoord, \
    RandomGammaTransfer, RandomAddLight
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataAug import z_score_clip

class Configuration:
    def __init__(self, normal_clip, train_cube_size, min_overlap, crop_lim, m_cover):
        self.normal_clip = normal_clip
        self.train_cube_size = train_cube_size
        self.min_overlap = min_overlap
        self.crop_lim = crop_lim
        self.m_cover = m_cover

def logCubenorm(logCube):
    logCube = np.where(logCube != -999, np.clip(logCube, a_min=3500, a_max=6000),logCube)
    MAX = np.max(logCube)
    MIN = np.min(np.where(logCube == -999, 9999999, logCube))
    logVec = np.where(logCube != -999, (logCube - MIN) / (MAX - MIN), logCube)
    print(MAX,MIN,np.min(logCube))
    return logVec,MAX,MIN

class SemiDataset(Dataset):
    def __init__(self, args, iters, seismic_road, logCube_road, F3):
        self.seismic = z_score_clip(np.load(seismic_road), args.normal_clip)
        self.F3 = F3
        if F3 == True:
            self.logCube,self.MAX,self.MIN = logCubenorm(np.load(logCube_road))
        else :
            self.logCube = np.load(logCube_road)

        self.wellPosList = np.argwhere(np.mean(self.logCube, axis=0) != -999)
        print(self.wellPosList)

        # self.freq = int(args.batch_size / 2)
        self.cube_size = args.train_cube_size
        y = np.arange(0, self.seismic.shape[1], dtype=np.float32)
        x = np.arange(0, self.seismic.shape[2], dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x)
        self.grid_y, self.grid_x = grid_y.T, grid_x.T
        self.iters = iters
        self.m_cover = args.m_cover
        # crop_lim默认是（0.5，2）sam_min,sam_max  提示：n和lamda的计算
        self.sam_min, self.sam_max = int(self.cube_size * args.crop_lim[0]), int(self.cube_size * args.crop_lim[1])
        # self.base_width = args.base_width

    def __len__(self):
        return self.iters



    def get_pos_len(self, pos, seismic_len, crop_cube_s):
        left = pos
        right = seismic_len - pos - 1
        len_max = crop_cube_s - self.m_cover - 1
        if left <= right:
            lmax = np.where(left < len_max, left, len_max)
            left_len = random.randint(self.m_cover, lmax)
            return left_len, crop_cube_s - left_len
        else:
            rmax = np.where(right < len_max, right, len_max)
            right_len = random.randint(self.m_cover, rmax)

            return crop_cube_s - right_len, right_len

    # 生成随机数，要是0的话，wh的取值范围是[sam_min,cube_size],是1的话，wh的取值范围是[cube_size,sam_max]
    def random_wh(self):
        if random.random() < 0.05: return self.cube_size
        if random.randint(0, 1):
            wh = random.randint(self.sam_min, self.cube_size)
        else:
            wh = random.randint(self.cube_size, self.sam_max)
        return wh

    def RandomResizeCropCoord(self, logCoord):
        # H->y, W->x
        # cube_coord:(x_upper_left, y_upper_left, x_lower_right, y_lower_right)

        # cH,cW是地震数据的inline,crossline
        _, cH, cW = self.seismic.shape
        # logY,logX是井的坐标
        logY, logX = logCoord
        # cube_size默认是48
        rH, rW = self.cube_size, self.cube_size

        '''
        有标签区域
        '''
        h_start, h_end = self.get_pos_len(logY, cH, self.random_wh())
        w_start, w_end = self.get_pos_len(logX, cW, self.random_wh())

        seismic = self.seismic[:, logY - h_start:logY + h_end, logX - w_start:logX + w_end]
        logCube = self.logCube[:, logY - h_start:logY + h_end, logX - w_start:logX + w_end]

        grid_y = self.grid_y[logY - h_start:logY + h_end, logX - w_start:logX + w_end]
        grid_x = self.grid_x[logY - h_start:logY + h_end, logX - w_start:logX + w_end]

        '''
        有标签区域大小缩放，并重建标签坐标
        '''
        tline, nH, nW = seismic.shape
        seismic = torch.from_numpy(seismic)
        grid_y = torch.from_numpy(grid_y)
        grid_x = torch.from_numpy(grid_x)
        seismic = F.interpolate(seismic[None, None], (tline, rH, rW), mode='trilinear', align_corners=True)[0, 0]
        grid_y = F.interpolate(grid_y[None, None], (rW, rH), mode='bilinear', align_corners=True)[0, 0, :, :, None]
        grid_x = F.interpolate(grid_x[None, None], (rW, rH), mode='bilinear', align_corners=True)[0, 0, :, :, None]
        grid = torch.cat((grid_y, grid_x), dim=-1).permute((2, 0, 1))
        logPoses = np.argwhere(np.mean(logCube, axis=0) != -999).tolist()
        new_logCube = np.zeros((tline, rH, rW)).astype(np.float32) - 999.
        for pos in logPoses:
            logVec = logCube[:, pos[0], pos[1]]
            new_logCube[:, np.clip(int(round(pos[0] * rH / nH)), a_min=0, a_max=self.cube_size - 1),
            np.clip(int(round(pos[1] * rW / nW)), a_min=0, a_max=self.cube_size - 1)] = logVec

        return seismic[None].float(), torch.from_numpy(new_logCube[None]).float(), grid.float()


    def strong_dataAug(self, seismic):
        seismic = RandomGammaTransfer(seismic)
        seismic = RandomNoise(seismic)
        seismic = RandomAddLight(seismic)
        return seismic

    def weak_dataAug(self, seismic, logCube, grid):
        seismic, logCube, grid = RandomHorizontalFlipCoord(seismic, logCube, grid, p=0.5)
        seismic, logCube, grid = RandomVerticalFlipCoord(seismic, logCube, grid, p=0.5)
        seismic, logCube, grid = RandomRotateCoord(seismic, logCube, grid, p=0.5)
        return seismic, logCube, grid



    def __getitem__(self, index):
        if self.F3 == True:
            wellPos = self.wellPosList[random.randint(5, len(self.wellPosList) - 1)]# 
        else:
            wellPos = self.wellPosList[random.randint(0, len(self.wellPosList) - 1)]

        seismic, logCube, _= self.RandomResizeCropCoord(wellPos)
        seismic_w, logCube, _ = self.weak_dataAug(seismic, logCube, _)
        seismic_s = self.strong_dataAug(seismic_w)
        return seismic_w, seismic_s, logCube

if __name__ == "__main__":
    config = Configuration(normal_clip=3.2, train_cube_size=48, min_overlap=6, crop_lim=[0.5,2], m_cover=3)

    train_set = SemiDataset(config, iters=10000, seismic_road='data/denoise_seismic.npy', logCube_road='data/logCube_16.npy',F3=False)

    train_loader = DataLoader(dataset=train_set, batch_size=2)
    for idx, batch in enumerate(train_loader):
        seismic, logCube, ul_seismic = batch