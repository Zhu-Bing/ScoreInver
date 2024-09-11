import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(q,k,v):
    d_k = k.size(-1)
    scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    return torch.matmul(p_attn, v)
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
class Generator(nn.Module):
    """Fake Label Generator in SemiReward"""

    def __init__(self, feature_dim=32):
        super(Generator, self).__init__()
        # self.fc_layers = nn.Sequential(
        # nn.Linear(feature_dim, feature_dim // 2),
        # nn.ReLU(),
        # nn.Linear(feature_dim // 2, feature_dim // 4),
        # nn.ReLU(),
        # nn.Linear(feature_dim // 4, feature_dim // 8),
        # nn.ReLU(),
        # nn.Linear(feature_dim // 8, feature_dim // 16),
        # nn.ReLU(),
        # nn.Linear(feature_dim // 16, feature_dim // 32)
        # )
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            # nn.ReLU(),
            # nn.Linear(feature_dim // 8, feature_dim // 16),
            # nn.ReLU(),
            # nn.Linear(feature_dim // 16, feature_dim // 32)
        )
        self._initialize_weights()


    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.fc_layers(x)
        x = F.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Rewarder(nn.Module):
    """Pseudo Label Reward in SemiReward"""

    def __init__(self, label_dim, label_embedding_dim, feature_dim=32):
        super(Rewarder, self).__init__()

        # Feature Processing Part
        self.feature_fc = nn.Linear(feature_dim, 1)
        self.feature_norm = nn.LayerNorm(label_embedding_dim)

        # Label Embedding Part
        self.label_embedding = nn.Linear(label_dim, label_embedding_dim)
        self.label_norm = nn.LayerNorm(label_embedding_dim)

        # Cross-Attention Mechanism
        self.cross_attention_fc = nn.Linear(label_embedding_dim, 1)

        # MLP (Multi-Layer Perceptron)
        self.mlp_fc1 = nn.Linear(400, 800)
        self.mlp_fc2 = nn.Linear(800, 400)
        self.mlp_norm = nn.LayerNorm(400)

        # Feed-Forward Network (FFN)
        self.ffn_fc1 = nn.Linear(400, 64)
        # self.ffn_norm = nn.LayerNorm(3200)
        self.ffn_fc2 = nn.Linear(64, 1)

    def forward(self, features, label):
        features = features.permute(0,2,1)
        features = self.feature_fc(features)
        features = features.squeeze(-1)
        # features = self.feature_norm(features)
        # Process Labels
        label_embed = self.label_embedding(label)
        label_embed = self.label_norm(label_embed)


        # Cross-Attention Mechanism
        cross_attention_input = torch.cat((features, label_embed), dim=0)
        cross_attention_weights = torch.softmax(self.cross_attention_fc(cross_attention_input), dim=0)
        cross_attention_output = (cross_attention_weights * cross_attention_input).sum(dim=0)

        # MLP Part
        mlp_input = torch.add(cross_attention_output.unsqueeze(0).expand(label_embed.size(0), -1), label_embed)
        mlp_output = F.relu(self.mlp_fc1(mlp_input))
        mlp_output = self.mlp_fc2(mlp_output)
        # mlp_output = self.mlp_norm(mlp_output)
        # FFN Part
        ffn_output = F.relu(self.ffn_fc1(mlp_output))
        # ffn_output = self.ffn_norm(ffn_output)
        ffn_output = self.ffn_fc2(ffn_output)
        reward = torch.sigmoid(ffn_output)
        return reward


class EMARewarder(Rewarder):
    """EMA version of Reward in SemiReward"""

    def __init__(self, label_dim, label_embedding_dim, feature_dim=384, ema_decay=0.9):
        super(EMARewarder, self).__init__(
            label_dim=label_dim, label_embedding_dim=label_embedding_dim, feature_dim=feature_dim)

        # EMA decay rate
        self.ema_decay = ema_decay

        # Initialize EMA parameters
        self.ema_params = {}
        self.initialize_ema()

    def initialize_ema(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = nn.Parameter(param.data.clone())

    def update_ema(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                ema_param = self.ema_params[name]
                if ema_param.device != param.device:
                    ema_param.data = param.data.clone().to(ema_param.device)
                else:
                    ema_param.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param.data)

    def forward(self, features, label_indices):
        # Process Features
        features = self.feature_fc(features)
        features = self.feature_norm(features)
        # Process Labels
        label_embed = self.label_embedding(label_indices)
        label_embed = self.label_norm(label_embed)
        # Cross-Attention Mechanism
        cross_attention_input = torch.cat((features, label_embed), dim=0)
        cross_attention_weights = torch.softmax(self.cross_attention_fc(cross_attention_input), dim=0)
        cross_attention_output = (cross_attention_weights * cross_attention_input).sum(dim=0)

        # MLP Part
        mlp_input = torch.add(cross_attention_output.unsqueeze(0).expand(label_embed.size(0), -1), label_embed)
        mlp_output = F.relu(self.mlp_fc1(mlp_input))
        mlp_output = self.mlp_fc2(mlp_output)

        # FFN Part
        ffn_output = F.relu(self.ffn_fc1(mlp_output))
        reward = torch.sigmoid(self.ffn_fc2(ffn_output))

        # Update EMA parameters
        self.update_ema()

        return reward


def cosine_similarity_n(x, y):
    # Calculate cosine similarity along the last dimension (dim=-1)
    cosine_similarity = torch.cosine_similarity(x, y, dim=-1, eps=1e-8)

    # Reshape the result to [first size of x, 1]
    normalized_similarity = (cosine_similarity + 1) / 2
    normalized_similarity = normalized_similarity.view(x.size(0), 1)

    return normalized_similarity


def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn_like(tensor) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor


def label_dim(x, default_dim=100):
    return int(max(default_dim, x))