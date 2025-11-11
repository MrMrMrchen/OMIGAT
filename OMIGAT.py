import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Parameter, ModuleList, Sequential, ReLU, Dropout
from torch_geometric.nn import TransformerConv, SAGEConv
from torch_geometric.utils import dropout_adj, degree
from torch_geometric.data import Data
import math


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


class RandomMasking(torch.nn.Module):

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.rand_like(x) > self.p
            x = x * mask
        return x


class FeatureWiseAttention(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.feature_attn_proj = Linear(in_channels, in_channels)
        self.feature_attn_score = Linear(in_channels, 1)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device
        eps = 1e-8

        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        deg = deg / (deg.max() + eps)
        clustering_coef = torch.sqrt(deg) / (torch.sqrt(deg).max() + eps)

        original_features = x
        deg_enhanced = original_features * deg.unsqueeze(1)
        cluster_enhanced = original_features * clustering_coef.unsqueeze(1)

        candidates = torch.stack([original_features, deg_enhanced, cluster_enhanced], dim=1)

        N, K, C = candidates.shape
        proj = torch.tanh(self.feature_attn_proj(candidates.view(N * K, C)))
        scores = self.feature_attn_score(proj).view(N, K, 1)
        attn_weights = F.softmax(scores, dim=1)

        fused_features = (attn_weights * candidates).sum(dim=1)
        return fused_features


class InteractionAttention(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.query = Linear(hidden_channels, hidden_channels)
        self.key = Linear(hidden_channels, hidden_channels)
        self.value = Linear(hidden_channels, hidden_channels)
        self.proj = Linear(hidden_channels, hidden_channels)
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.view(-1, self.hidden_channels)
        if x2.dim() > 2:
            x2 = x2.view(-1, self.hidden_channels)

        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_channels)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = self.proj(out)
        out = x1 + out
        out = self.norm(out)

        return out

class CustomSAGEConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.sage_conv(x, edge_index)


class OMIGATNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels=128, num_layers=2, num_heads=4,
                 dropout=0.2, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.feature_wise_attention = FeatureWiseAttention(in_channels)
        self.random_masking = RandomMasking(p=0.2)
        self.linear_proj = Linear(in_channels, hidden_channels)

        self.sage_layers = ModuleList([
            CustomSAGEConv(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])

        self.transformer_convs_prior = ModuleList([
            TransformerConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout, edge_dim=None, beta=True
            ) for _ in range(num_layers)
        ])

        self.transformer_convs_knn = ModuleList([
            TransformerConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout, edge_dim=None, beta=True
            ) for _ in range(num_layers)
        ])

        self.interaction_attention = InteractionAttention(hidden_channels)

        self.sage_proj = Linear(hidden_channels, hidden_channels)
        self.global_proj = Linear(hidden_channels, hidden_channels)
        self.lambda_param = Parameter(torch.tensor([0.5], dtype=torch.float32))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1)
        )

        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

    def compute_loss(self, pred, target):
        return focal_loss(pred, target, self.focal_alpha, self.focal_gamma)

    def build_knn_graph(self, features, k=10):
        from torch_geometric.nn import knn_graph
        edge_index_knn = knn_graph(features, k=k, loop=False)
        return edge_index_knn

    def forward(self, data):
        x, edge_index_prior = data.x, data.edge_index

        if hasattr(data, 'edge_index_knn'):
            edge_index_knn = data.edge_index_knn
        else:
            edge_index_knn = self.build_knn_graph(x, k=10)

        x_fused = self.feature_wise_attention(x, edge_index_prior)
        x_masked = self.random_masking(x_fused)
        H = self.linear_proj(x_masked)

        H_sage = H
        for sage_layer in self.sage_layers:
            H_sage = F.relu(sage_layer(H_sage, edge_index_prior))
            H_sage = F.dropout(H_sage, p=self.dropout, training=self.training)

        R0 = self.norm1(F.gelu(H))

        edge_index_prior_drop, _ = dropout_adj(edge_index_prior, p=0.1,
                                               force_undirected=True,
                                               num_nodes=H.shape[0],
                                               training=self.training)
        edge_index_knn_drop, _ = dropout_adj(edge_index_knn, p=0.1,
                                             force_undirected=True,
                                             num_nodes=H.shape[0],
                                             training=self.training)

        R_prior = R0
        R_knn = R0

        for trans_prior, trans_knn in zip(self.transformer_convs_prior, self.transformer_convs_knn):
            R_prior = trans_prior(R_prior, edge_index_prior_drop)
            R_knn = trans_knn(R_knn, edge_index_knn_drop)

        R_global_1 = self.interaction_attention(R_prior, R_knn)
        R_global_2 = self.interaction_attention(R_knn, R_prior)
        R_global = R_global_1 + R_global_2
        R_global = self.norm2(F.relu(R_global))
        R_global = F.dropout(R_global, p=self.dropout, training=self.training)

        H_sage_proj = self.sage_proj(H_sage)
        R_global_proj = self.global_proj(R_global)

        lambda_weight = torch.sigmoid(self.lambda_param)
        z = lambda_weight * R_global_proj + (1 - lambda_weight) * H_sage_proj

        y_hat = self.mlp(z)

        return y_hat


if __name__ == "__main__":

    in_channels = 100
    hidden_channels = 128
    num_layers = 2

    model = OMIGATNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    )

    num_nodes = 1000
    x = torch.randn(num_nodes, in_channels)
    edge_index_prior = torch.randint(0, num_nodes, (2, 2000))
    edge_index_knn = torch.randint(0, num_nodes, (2, 3000))

    data = Data(x=x, edge_index=edge_index_prior, edge_index_knn=edge_index_knn)

    output = model(data)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")