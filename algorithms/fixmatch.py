import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from torch.cuda import amp
from utils.loss_class import lambda_loss
from algorithms.semireward import Generator,Rewarder
def find_valid_coordinates(data):
    for a in range(data.size(1)):
        for b in range(data.size(2)):
            # 计算当前 (a, b) 坐标处的平均值
            avg_value = torch.mean(data[:, a, b])
            # 检查平均值是否不为 -999
            if avg_value != -999:
                return a, b  # 返回第一个满足条件的坐标 (a, b)
    return None  # 如果找不到满足条件的坐标，则返回 None
class FixMatch(nn.Module):
    def __init__(self,arg,backbone):
        super().__init__()
        self.backbone = backbone
        self.batch_size = arg.batch_size
        self.start_timing = arg.start_timing
        self.loss_function = lambda_loss()
        self.loss_flag = arg.loss
        self.rewarder = Rewarder(400,1600)
        self.generator = Generator()
        self.rewarder_optimizer = torch.optim.Adam(self.rewarder.parameters(), lr=arg.sr_lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=arg.sr_lr)
        self.criterion = torch.nn.MSELoss()
    def data_generator(self, x, y_lb, rewarder):
        rewarder = rewarder.eval()

        for _ in range(self.sr_decay()):
            num_lb = y_lb.shape[0]
            with self.amp_cm():
                if self.use_cat:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    outputs = self.model(inputs)
                    logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                    feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)


                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                      logits=probs_x_ulb_w,
                                      use_hard_label=self.use_hard_label,
                                      T=self.T,
                                      softmax=False)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
            reward = rewarder(feats_x_ulb_w, pseudo_label)
            avg_reward=reward.mean()
            mask2 = torch.where(reward >= avg_reward, torch.tensor(1), torch.tensor(0)).squeeze().float()
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label,'ce', mask=mask,mask2=mask2)
            unsup_loss = self.loss_function(logits_x_lb, logCube, self.loss_flag)

    def forward(self, x, logCube, epoch):
        outputs, x_togenerater, x_torewarder = self.backbone(x,logCube)
        '''
        三个数据体
        '''
        logits_x_lb = outputs[:self.batch_size]
        logits_x_ulb_w, logits_x_ulb_s = outputs[self.batch_size:].chunk(2)
        probs_x_ulb_w = logits_x_ulb_w.detach()
        togenerater_x_lb = x_togenerater[:self.batch_size]
        togenerater_x_ulb_w, togenerater_x_ulb_s = x_togenerater[self.batch_size:].chunk(2)

        torewarder_x_lb = x_torewarder[:self.batch_size]
        torewarder_x_ulb_w, torewarder_x_ulb_s = x_torewarder[self.batch_size:].chunk(2)

        torewarder_x_lb = torewarder_x_lb.flatten(start_dim=1)
        # torewarder_x_ulb_w = torewarder_x_ulb_w.flatten(start_dim=1)
        # torewarder_x_ulb_s = torewarder_x_ulb_s.flatten(start_dim=1)
        sup_loss = self.loss_function(logits_x_lb, logCube, self.loss_flag)

        # if epoch >= self.start_timing:
        #     rewarder = self.rewarder
        #     unsup_loss = self.data_generator(x, logCube, rewarder)
        # else:
        #     unsup_loss = 0


        if epoch > 0:
            self.rewarder.train()
            self.generator.train()
            real_labels =[]
            generated_labels = []
            generatecube = self.generator(togenerater_x_lb.detach()).detach()

            for i in range(self.batch_size):
                log = logCube[i, :][0]
                coordinates_x, coordinates_y = find_valid_coordinates(log)
                real_labels.append(logCube[i, :, :, coordinates_x, coordinates_y])
                generated_labels.append(generatecube[i, :, :, coordinates_x, coordinates_y])
            real_labels = torch.cat(real_labels,dim=0)
            generated_labels = torch.cat(generated_labels,dim=0)
            reward = self.rewarder(torewarder_x_lb.detach(), generated_labels.squeeze(1))
            if epoch < self.start_timing:
                cosine_similarity_score = (F.cosine_similarity(generated_labels.float(), real_labels.float(), dim=1) + 1) / 2
                cosine_similarity_score = cosine_similarity_score.view(generated_labels.size(0), 1)
                generator_loss = self.criterion(reward, torch.ones_like(reward))
                rewarder_loss = self.criterion(reward, cosine_similarity_score)

                self.generator_optimizer.zero_grad()
                self.rewarder_optimizer.zero_grad()

                generator_loss.backward(retain_graph=True)
                rewarder_loss.backward(retain_graph=True)

                self.generator_optimizer.step()
                self.rewarder_optimizer.step()

        total_loss = sup_loss + unsup_loss
        return total_loss,sup_loss,unsup_loss





