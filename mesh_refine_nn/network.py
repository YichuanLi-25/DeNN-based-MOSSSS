import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

class MeshUpdateNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.encoder = geom_nn.Sequential('x, pos', [
            (geom_nn.EdgeConv(nn.Sequential(
                nn.Linear(2*input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ), 'x, pos'), 'x -> x'),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3))  # 输出坐标偏移量

    def forward(self, data):
        offsets = self.decoder(self.encoder(data.x, data.pos))
        return data.pos + 0.1 * torch.tanh(offsets)  # 限制初始变化幅度