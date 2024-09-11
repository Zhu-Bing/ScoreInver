import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# AI = np.load("data/F3/AI.npy")
# AI = AI[:,0:600,:]
# print(AI.shape)
F3fixmatch1 = np.load("E:\朱信源\project\F3fixmatch2750(1).npy")
print(F3fixmatch1.shape)
F3fixmatch2 = np.load("E:\朱信源\project\F3fixmatch2750(2).npy")
F3fixmatch = np.concatenate((F3fixmatch1,F3fixmatch2),axis=-2)
print(F3fixmatch.shape)

F3fixmatch = torch.from_numpy(F3fixmatch)
F3fixmatch = F.interpolate(F3fixmatch[None,None], (400, 600, 951), mode='trilinear', align_corners=True)
F3fixmatch = F3fixmatch[0,0,:]
F3fixmatch = F3fixmatch.detach().numpy()
np.save("F3fixmatch2750.npy", F3fixmatch)
print(F3fixmatch.shape)
