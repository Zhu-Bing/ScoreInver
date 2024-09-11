import torch
import torch.nn as nn
import torch.nn.functional as F


class Rewarder(nn.Module):
    """Pseudo Label Reward in SemiReward"""

    def __init__(self, label_dim=5, label_embedding_dim=128, feature_dim=5):
        super(Rewarder, self).__init__()

        # Feature Processing Part
        self.feature_fc = nn.Linear(feature_dim, 128)
        self.feature_norm = nn.LayerNorm(128)

        # Label Embedding Part
        self.label_embedding = nn.Embedding(label_dim, label_embedding_dim)
        self.label_norm = nn.LayerNorm(label_embedding_dim)

        # Cross-Attention Mechanism
        self.cross_attention_fc = nn.Linear(128, 1)

        # MLP (Multi-Layer Perceptron)
        self.mlp_fc1 = nn.Linear(128, 256)
        self.mlp_fc2 = nn.Linear(256, 128)

        # Feed-Forward Network (FFN)
        self.ffn_fc1 = nn.Linear(128, 64)
        self.ffn_fc2 = nn.Linear(64, 1)

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
        return reward


rewarder = Rewarder()
generated_label = torch.tensor([[1], [2], [3], [4]])
real_labels_tensor = torch.tensor([[0], [1], [2], [3]])
batch_size = 4
num_classes = 5
# 转换为独热编码
generated_label = F.one_hot(generated_label.squeeze(1), num_classes=num_classes)
# real_labels_tensor = F.one_hot(real_labels_tensor, num_classes=num_classes)
real_labels_tensor = real_labels_tensor.squeeze(1)
reward = rewarder(generated_label.float(),real_labels_tensor.long())