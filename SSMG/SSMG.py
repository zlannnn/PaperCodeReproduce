import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationSensitiveAttention(nn.Module):
    def __init__(self, feature_size, heads):
        super(LocationSensitiveAttention, self).__init__()
        self.feature_size = feature_size  # 特征大小
        self.heads = heads  # 注意力头的数量
        self.head_dim = feature_size // heads  # 每个头的维度

        assert (
            self.head_dim * heads == feature_size
        ), "特征大小需要能被头的数量整除"

        # 初始化交叉注意力的键、查询和值的线性层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # 初始化门控机制的可学习参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, noise_features, spatial_semantic_map):
        N = noise_features.shape[0]  # 批量大小
        value_len, key_len, query_len = noise_features.shape[1], noise_features.shape[1], spatial_semantic_map.shape[1]

        # 将嵌入分成多个头
        values = self.values(noise_features).reshape(N, value_len, self.heads, self.head_dim)
        keys = self.keys(noise_features).reshape(N, key_len, self.heads, self.head_dim)
        queries = self.queries(spatial_semantic_map).reshape(N, query_len, self.heads, self.head_dim)

        # 应用缩放因子进行点积
        attention = torch.einsum('nqhd,nkhd->nhqk', [queries, keys]) / torch.sqrt(torch.FloatTensor([self.head_dim]))

        # 应用softmax获取注意力权重
        attention = F.softmax(attention, dim=-1)

        # 将注意力权重应用到值上
        out = torch.einsum('nhqk,nlhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        # 通过tanh和可学习参数γ调节门控机制
        out = self.gamma * torch.tanh(out)

        # 将门控后的输出加到原始噪声特征上
        return noise_features + out


class RelationSensitiveAttention(nn.Module):
    def __init__(self, feature_size, heads):
        super(RelationSensitiveAttention, self).__init__()
        self.feature_size = feature_size  # 特征大小
        self.heads = heads  # 注意力头的数量
        self.head_dim = feature_size // heads  # 每个头的维度
        assert self.head_dim * heads == feature_size, "特征大小需要能被头的数量整除"
        
        # 实例到实例和实例到场景的线性变换
        self.query_proj = nn.Linear(feature_size, feature_size)  # 查询的线性变换
        self.key_proj = nn.Linear(feature_size, feature_size)  # 键的线性变换
        self.value_proj = nn.Linear(feature_size, feature_size)  # 值的线性变换
        self.fc_out = nn.Linear(feature_size, feature_size)  # 输出的全连接层

    def forward(self, feature_map, scene_token, relation_matrix):
        batch_size, num_tokens, feature_size = feature_map.size()
        
        # 添加全局场景标记到特征图
        scene_token = scene_token.unsqueeze(1).expand(-1, num_tokens, -1)
        feature_map_with_scene = torch.cat((scene_token, feature_map), dim=1)

        # 计算查询、键、值
        queries = self.query_proj(feature_map_with_scene)
        keys = self.key_proj(feature_map_with_scene)
        values = self.value_proj(feature_map_with_scene)
        
        # 调整形状以支持多头注意力
        queries = queries.view(batch_size, -1, self.heads, self.head_dim)
        keys = keys.view(batch_size, -1, self.heads, self.head_dim)
        values = values.view(batch_size, -1, self.heads, self.head_dim)

        # 计算注意力分数
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', queries, keys)
        attention_scores = attention_scores / torch.sqrt(torch.FloatTensor([self.head_dim]))
        
        # 使用关系矩阵调整注意力分数
        attention_scores = attention_scores + relation_matrix.unsqueeze(0).unsqueeze(0)
        
        # 应用softmax并计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重到值上
        attention_output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, values)
        
        # 调整形状并应用输出的全连接层
        attention_output = attention_output.contiguous().view(batch_size, -1, feature_size)
        output = self.fc_out(attention_output)
        
        # 返回注意力机制的输出
        return output