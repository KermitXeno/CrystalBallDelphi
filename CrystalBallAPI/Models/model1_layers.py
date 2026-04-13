import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):

    def __init__(self, d_in, d_out, d_ctx = None, dropout = 0.1):
        super().__init__()
        self.proj_skip = nn.Linear(d_in, d_out, bias = False) if d_in != d_out else nn.Identity()
        ctx_dim = d_ctx or 0
        self.fc_hidden = nn.Linear(d_in + ctx_dim, d_out)
        self.fc_gate = nn.Linear(d_in + ctx_dim, d_out)
        nn.init.constant_(self.fc_gate.bias, -2.0)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx = None):
        h = torch.cat([x, ctx], dim = -1) if ctx is not None else x
        gate = torch.sigmoid(self.fc_gate(h))
        act = F.elu(self.fc_hidden(h))
        return self.norm(self.proj_skip(x) + self.drop(gate * act))


class CausalConv1d(nn.Module):

    def __init__(self, d_in, d_out, kernel_size, dilation = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_in, d_out, kernel_size, dilation = dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalEdgeEncoder(nn.Module):

    def __init__(self, F_edge, d_edge, dropout = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(F_edge, d_edge, kernel_size = 3, padding = 0)
        self.conv2 = nn.Conv1d(d_edge, d_edge, kernel_size = 3, padding = 0)
        self.temporal_attn = nn.Linear(d_edge, 1, bias = False)
        self.norm = nn.LayerNorm(d_edge)
        self.drop = nn.Dropout(dropout)

    def forward(self, edge_features):
        B, k, N, _, Fe = edge_features.shape
        x = edge_features.permute(0, 2, 3, 4, 1).reshape(B * N * N, Fe, k)
        x = F.pad(x, (2, 0))
        x = F.elu(self.conv1(x))
        x = F.pad(x, (2, 0))
        x = self.conv2(x)
        x = x.transpose(1, 2)
        w = F.softmax(self.temporal_attn(x), dim = 1)
        x = (w * x).sum(dim = 1)
        return self.drop(self.norm(x.view(B, N, N, -1)))


class EdgeBiasNet(nn.Module):

    def __init__(self, F_edge, n_heads):
        super().__init__()
        self.net = nn.Linear(F_edge, n_heads, bias = True)
        nn.init.normal_(self.net.weight, std = 0.01)
        nn.init.zeros_(self.net.bias)

    def forward(self, edge_features):
        bias = self.net(edge_features)
        return bias.permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):

    def __init__(self, drop_prob = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device = x.device))
        return x * mask / keep


class GrangerAttention(nn.Module):

    def __init__(self, d_in, d_out, n_heads = 4, F_edge = 4,
                 dropout = 0.1, edge_drop = 0.1):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(d_in, d_out, bias = False)
        self.W_k = nn.Linear(d_in, d_out, bias = False)
        self.W_v = nn.Linear(d_in, d_out, bias = False)
        self.W_o = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.edge_bias = EdgeBiasNet(F_edge, n_heads)
        self.edge_drop = edge_drop

    def forward(self, x, edge_features):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        edge_bias = self.edge_bias(edge_features).transpose(-2, -1)
        if self.training and self.edge_drop > 0:
            mask = torch.bernoulli(torch.full_like(edge_bias, 1.0 - self.edge_drop))
            edge_bias = edge_bias * mask / (1.0 - self.edge_drop)
        scores = scores + edge_bias

        attn = self.dropout(F.softmax(scores, dim = -1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)


class GrangerGraphEncoder(nn.Module):

    def __init__(self, F_node, d_embed, n_heads = 4, n_layers = 2,
                 F_edge = 4, d_edge = 16, dropout = 0.1,
                 stoch_depth = 0.1, edge_drop = 0.1, k_graph = 8):
        super().__init__()
        self.temporal_edge_enc = TemporalEdgeEncoder(F_edge, d_edge, dropout = dropout)
        self.input_proj = nn.Linear(F_node, d_embed)
        self.time_embed = nn.Parameter(torch.randn(1, k_graph, 1, d_embed) * 0.01)

        self.node_temporal_conv1 = CausalConv1d(d_embed, d_embed, kernel_size = 3)
        self.node_temporal_conv2 = CausalConv1d(d_embed, d_embed, kernel_size = 3)
        self.node_temporal_norm = nn.LayerNorm(d_embed)
        self.node_temporal_drop = nn.Dropout(dropout)

        self.attn_layers = nn.ModuleList([
            GrangerAttention(d_embed, d_embed, n_heads = n_heads,
                             F_edge = d_edge, dropout = dropout,
                             edge_drop = edge_drop)
            for _ in range(n_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in range(n_layers)])

        drop_rates = [stoch_depth * (i + 1) / n_layers for i in range(n_layers)]
        self.stoch_depths = nn.ModuleList([StochasticDepth(p) for p in drop_rates])

        self.ffn_layers = nn.ModuleList([
            GRN(d_embed, d_embed, dropout = dropout) for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in range(n_layers)])

        self.global_grn = GRN(d_embed, d_embed, d_ctx = d_embed, dropout = dropout)
        self.global_norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_features, asset_embed = None):
        B, k, N, Fn = node_features.shape
        d = self.input_proj.out_features

        h = self.input_proj(node_features.reshape(B * k * N, Fn))
        h = h.view(B, k, N, d)

        te = self.time_embed
        if te.shape[1] != k:
            te = te[:, -k:, :, :] if te.shape[1] > k else F.pad(te, (0, 0, 0, 0, k - te.shape[1], 0))
        h = h + te

        if asset_embed is not None:
            h = h + asset_embed[None, None, :, :]

        ht = h.permute(0, 2, 3, 1).reshape(B * N, d, k)
        ht = F.elu(self.node_temporal_conv1(ht))
        ht = self.node_temporal_conv2(ht)
        ht = ht[:, :, -1].view(B, N, d)

        h_last = h[:, -1, :, :]
        h = self.node_temporal_norm(self.node_temporal_drop(ht) + h_last)

        edge_enc = self.temporal_edge_enc(edge_features)
        for attn, a_norm, sd, ffn, f_norm in zip(
                self.attn_layers, self.attn_norms, self.stoch_depths,
                self.ffn_layers, self.ffn_norms):
            h = a_norm(h + sd(self.dropout(attn(h, edge_enc))))
            h = f_norm(h + self.dropout(ffn(h)))

        global_ctx = h.mean(dim = 1, keepdim = True).expand(-1, h.shape[1], -1)
        h = self.global_norm(h + self.dropout(self.global_grn(h, ctx = global_ctx)))
        return h


class TemporalAssetEncoder(nn.Module):

    def __init__(self, F_node, F_global, D_time, d_model = 64, d_lstm = 128,
                 n_lstm_layers = 1, pool_stride = 4, dropout = 0.1, n_pool_heads = 4):
        super().__init__()
        self.pool_stride = pool_stride
        self.n_pool_heads = n_pool_heads
        self.node_proj = nn.Linear(F_node, d_model)
        self.pool_attn = nn.Linear(d_model, n_pool_heads, bias = False)
        self.pool_merge = nn.Linear(d_model * n_pool_heads, d_model)

        d_step = d_model + F_global + D_time

        self.conv_local = CausalConv1d(d_step, d_step, kernel_size = 3, dilation = 1)
        self.conv_mid = CausalConv1d(d_step, d_step, kernel_size = 3, dilation = 4)
        self.conv_wide = CausalConv1d(d_step, d_step, kernel_size = 3, dilation = 16)
        self.scale_norm = nn.LayerNorm(d_step)
        self.scale_proj = nn.Linear(d_step * 3, d_step)

        self.downsample = nn.AvgPool1d(kernel_size = pool_stride, stride = pool_stride)

        self.input_grn = GRN(d_step, d_step, dropout = dropout)
        self.lstm = nn.LSTM(
            input_size = d_step,
            hidden_size = d_lstm,
            num_layers = n_lstm_layers,
            batch_first = True,
            dropout = dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_out_drop = nn.Dropout(dropout)
        self.temporal_query = nn.Linear(d_lstm, 1, bias = False)
        self.out_norm = nn.LayerNorm(d_lstm)
        self.d_lstm = d_lstm

    def forward(self, node_features, global_features, time_enc, asset_embed = None):
        B, T, N, Fn = node_features.shape
        node_emb = self.node_proj(node_features.view(B * T, N, Fn))

        if asset_embed is not None:
            node_emb = node_emb + asset_embed[None, :, :]

        a_w = F.softmax(self.pool_attn(node_emb), dim = 1)
        node_pool = torch.einsum("bnh,bnd->bhd", a_w, node_emb)
        node_pool = self.pool_merge(node_pool.reshape(B * T, -1)).view(B, T, -1)

        step_in = torch.cat([node_pool, global_features, time_enc], dim = -1)
        x = step_in.transpose(1, 2)

        s_local = F.elu(self.conv_local(x))
        s_mid = F.elu(self.conv_mid(x))
        s_wide = F.elu(self.conv_wide(x))
        merged = self.scale_proj(torch.cat([s_local, s_mid, s_wide], dim = 1).transpose(1, 2))
        merged = self.scale_norm(merged + step_in)

        T_pad = T - (T % self.pool_stride)
        if T_pad < T:
            merged = merged[:, T - T_pad :, :]
        ds = self.downsample(merged.transpose(1, 2)).transpose(1, 2)

        lstm_in = self.input_grn(ds)
        lstm_out, (h_n, _) = self.lstm(lstm_in)
        lstm_out = self.lstm_out_drop(lstm_out)

        t_w = F.softmax(self.temporal_query(lstm_out), dim = 1)
        attn_pool = (t_w * lstm_out).sum(dim = 1)

        return self.out_norm(h_n[-1] + attn_pool)


class SentimentEncoder(nn.Module):

    def __init__(self, N, F_sentiment, d_out, dropout = 0.1):
        super().__init__()
        self.per_asset_grn = GRN(F_sentiment, d_out, dropout = dropout)
        self.pool_attn = nn.Linear(d_out, 1, bias = False)
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, sentiment_scores, sentiment_missing):
        B = sentiment_scores.shape[0]
        gate = (1.0 - sentiment_missing).view(B, 1, 1)
        per_asset = self.per_asset_grn(sentiment_scores * gate)
        w = F.softmax(self.pool_attn(per_asset), dim = 1)
        emb = self.out_norm((w * per_asset).sum(dim = 1))
        return torch.cat([emb, sentiment_missing.unsqueeze(-1)], dim = -1)


class CosineClassifier(nn.Module):

    def __init__(self, d_in, n_classes, init_temp = 10.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_in)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_in) * 0.01)
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())

    def forward(self, x):
        h = F.gelu(self.proj(x))
        h_norm = F.normalize(h, dim = -1)
        p_norm = F.normalize(self.prototypes, dim = -1)
        return self.log_temp.exp() * (h_norm @ p_norm.T)


class SupConRegularizer(nn.Module):

    def __init__(self, d_in, d_proj = 64, temperature = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.ReLU(),
            nn.Linear(d_in // 2, d_proj),
        )
        self.temperature = temperature

    def forward(self, features, labels):
        z = F.normalize(self.proj(features), dim = 1)
        B = z.size(0)
        sim = z @ z.T / self.temperature

        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        pos_count = mask.sum(dim = 1).clamp(min = 1)

        logits_max = sim.max(dim = 1, keepdim = True).values.detach()
        logits = sim - logits_max
        exp_logits = torch.exp(logits)
        diag_mask = 1.0 - torch.eye(B, device = exp_logits.device)
        exp_logits = exp_logits * diag_mask
        log_prob = logits - torch.log(exp_logits.sum(dim = 1, keepdim = True) + 1e-9)

        return -(mask * log_prob).sum(dim = 1).div(pos_count).mean()


class RDropKL(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits_a, logits_b):
        p = F.log_softmax(logits_a, dim = -1)
        q = F.log_softmax(logits_b, dim = -1)
        kl_pq = F.kl_div(q, p.exp(), reduction = "batchmean")
        kl_qp = F.kl_div(p, q.exp(), reduction = "batchmean")
        return (kl_pq + kl_qp) * 0.5


class EMAModel:

    def __init__(self, model, decay = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return dict(self.shadow)

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}
