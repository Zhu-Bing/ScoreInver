# from skimage import transform as sk_trans
import numpy as np
import random
import torch
from torchvision.transforms import functional as F


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def z_score_clip(data, clp_s=3.2):
    z = (data - np.mean(data)) / np.std(data)
    # print(np.max(data), np.min(data))
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))


def randomCrop(data, size=(128, 128, 128)):
    shape = data[0].shape
    size = np.array(size)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    c = random.randint(0, lim[2])
    return [d[w:w + size[0], h:h + size[1], c:c + size[2]] for d in data]


# F.rotate()

def zscore_clip(data, clp=5):
    data = (data - np.mean(data)) / np.std(data)
    return np.clip(data, a_min=-clp, a_max=clp)


def RandomHorizontalFlipCoord(seismic, logCube, coord, p=0.5):
    if random.random() < p:
        return F.hflip(seismic), F.hflip(logCube), F.hflip(coord)
    return seismic, logCube, coord


# def random_mask(data, p=0.5):
#     if random.random() < p:
#         return data
#     _, t, h, w = data.shape
#     if random.randint(0, 1):
#         ty = random.randint(2, config.m_cover * 4)
#         tx = random.randint(2, config.m_cover * 4)
#         tyy = h - ty - 1
#         txx = w - tx - 1
#         bty = np.random.randint(0, tyy)
#         btx = np.random.randint(0, txx)
#         if random.randint(0, 1):
#             if random.randint(0, 1):
#                 data[:, :, bty:bty + ty, :] = 0.5
#             else:
#                 data[:, :, :, btx:btx + tx] = 0.5
#         else:
#             data[:, :, bty:bty + ty, btx:btx + tx] = 0.5
#     else:
#         prop = random.random() / 2
#         s = random.sample(range(0, h), int(h * prop))
#         if random.randint(0, 1):
#             for i in s: data[:, :, :, i] = 0.5
#         else:
#             for i in s: data[:, :, i, :] = 0.5
#
#     return data


def RandomVerticalFlipCoord(seismic, logCube, coord, p=0.5):
    if random.random() < p:
        return F.vflip(seismic), F.vflip(logCube), F.vflip(coord)
    return seismic, logCube, coord


def RandomRotateCoord(seismic, logCube, coord, p=0.5):
    if random.random() < p:
        return seismic.permute((0, 1, 3, 2)), logCube.permute((0, 1, 3, 2)), coord.permute((0, 2, 1))
    return seismic, logCube, coord


def RandomNoise(seismic):
    if random.random() < 0.15:  return seismic
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    scale = random.random() * (s_max - s_min) * 0.15
    noise = torch.normal(mean=0.0, std=scale, size=seismic.shape)
    if random.randint(0, 1):
        sigma = random.randint(1, 3)
        noise = F.gaussian_blur(noise, kernel_size=sigma * 2 + 1, sigma=sigma)
    seismic = seismic + noise
    # if random.randint(0,1):
    seismic = torch.clip(seismic, min=s_min, max=s_max)
    return seismic


def RandomAddLight(seismic):
    if random.random() < 0.15:  return seismic
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    scale = random.random() * (s_max - s_min) * 0.1
    if random.randint(0, 1):
        seismic = seismic + scale
    else:
        seismic = seismic - scale
    # seismic = seismic + scale
    seismic = torch.clip(seismic, min=s_min, max=s_max)
    return seismic


def RandomGammaTransfer(seismic):
    if random.random() < 0.15:  return seismic
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    gamma = np.random.normal(loc=1.0, scale=0.1)
    gamma_seismic = (seismic - s_min) ** gamma

    gamma_range = torch.max(gamma_seismic) - torch.min(gamma_seismic)
    gamma_seismic = ((gamma_seismic - torch.min(gamma_seismic)) / gamma_range) * (s_max - s_min) + s_min
    return gamma_seismic


def resize_time(seis, T, H, W):
    if random.random() < 0.2: return seis
    if random.randint(0, 1):
        time_scale = random.uniform(1, 2)
    else:
        time_scale = random.uniform(0.5, 1)
    resize_time = int(round(T * time_scale) + (16 - (round(T * time_scale) % 16)))
    seis = torch.nn.functional.interpolate(seis, (resize_time, H, W), mode='trilinear', align_corners=True)
    return seis
