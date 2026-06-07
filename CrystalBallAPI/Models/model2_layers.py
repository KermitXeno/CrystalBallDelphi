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


class SubHourlyEncoder(nn.Module):

    def __init__(self, F_in, D_time, d_out, dropout = 0.1):
        super().__init__()
        self.proj = nn.Linear(F_in + D_time, d_out)
        self.conv1 = CausalConv1d(d_out, d_out, kernel_size = 3, dilation = 1)
        self.conv2 = CausalConv1d(d_out, d_out, kernel_size = 3, dilation = 4)
        self.norm = nn.LayerNorm(d_out)
        self.attn = nn.Linear(d_out, 1, bias = False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, time_enc):
        B, T, N, Fin = x.shape
        te = time_enc.unsqueeze(2).expand(-1, -1, N, -1)
        h = self.proj(torch.cat([x, te], dim = -1))
        d = h.shape[-1]
        ht = h.permute(0, 2, 3, 1).reshape(B * N, d, T)
        ht = F.elu(self.conv1(ht))
        ht = self.conv2(ht)
        ht = ht.transpose(1, 2)
        w = F.softmax(self.attn(ht), dim = 1)
        pooled = (w * ht).sum(dim = 1).view(B, N, d)
        return self.drop(self.norm(pooled))


class AssetTemporalEncoder(nn.Module):

    def __init__(self, F_in, D_time, d_model = 32, d_lstm = 32, dropout = 0.1):
        super().__init__()
        self.proj = nn.Linear(F_in + D_time, d_model)
        self.conv_short = CausalConv1d(d_model, d_model, kernel_size = 3, dilation = 1)
        self.conv_mid = CausalConv1d(d_model, d_model, kernel_size = 3, dilation = 4)
        self.conv_long = CausalConv1d(d_model, d_model, kernel_size = 3, dilation = 12)
        self.scale_proj = nn.Linear(d_model * 3, d_model)
        self.scale_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_lstm, batch_first = True)
        self.out_norm = nn.LayerNorm(d_lstm)
        self.query = nn.Linear(d_lstm, 1, bias = False)
        self.drop = nn.Dropout(dropout)
        self.d_lstm = d_lstm

    def forward(self, x, time_enc, t_recent = 4):
        B, T, N, Fin = x.shape
        te = time_enc.unsqueeze(2).expand(-1, -1, N, -1)
        h = self.proj(torch.cat([x, te], dim = -1))
        d = h.shape[-1]
        ht = h.permute(0, 2, 3, 1).reshape(B * N, d, T)
        s1 = F.elu(self.conv_short(ht))
        s2 = F.elu(self.conv_mid(ht))
        s3 = F.elu(self.conv_long(ht))
        merged = self.scale_proj(torch.cat([s1, s2, s3], dim = 1).transpose(1, 2))
        merged = self.scale_norm(merged + ht.transpose(1, 2))
        lstm_out, (h_n, _) = self.lstm(merged)
        w = F.softmax(self.query(lstm_out), dim = 1)
        pooled = (w * lstm_out).sum(dim = 1)
        out = self.out_norm(h_n[-1] + pooled)
        out = self.drop(out).view(B, N, self.d_lstm)
        tr = min(t_recent, T)
        recent = lstm_out[:, -tr:, :].reshape(B, N, tr, self.d_lstm)
        return out, recent


class DynamicAdjacency(nn.Module):

    def __init__(self, N, n_heads, d_node, d_regime, dropout = 0.1, init_scale = 0.1):
        super().__init__()
        self.base_adj = nn.Parameter(torch.randn(n_heads, N, N) * init_scale)
        self.node_key = nn.Linear(d_node, n_heads, bias = False)
        self.regime_proj = nn.Linear(d_regime, n_heads, bias = False)
        self.drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        nn.init.zeros_(self.node_key.weight)
        nn.init.zeros_(self.regime_proj.weight)

    def forward(self, h, regime_ctx = None):
        nk = self.node_key(h)
        mod_i = nk.unsqueeze(2)
        mod_j = nk.unsqueeze(1)
        dynamic = (mod_i + mod_j).permute(0, 3, 1, 2)
        bias = self.base_adj.unsqueeze(0) + dynamic
        if regime_ctx is not None:
            r = self.regime_proj(regime_ctx)
            bias = bias + r.unsqueeze(-1).unsqueeze(-1)
        return self.drop(bias)


class CrossAssetAttention(nn.Module):

    def __init__(self, d_in, d_out, N, n_heads = 4, d_regime = 5, dropout = 0.1):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.scale = self.d_head ** -0.5
        self.W_q = nn.Linear(d_in, d_out, bias = False)
        self.W_k = nn.Linear(d_in, d_out, bias = False)
        self.W_v = nn.Linear(d_in, d_out, bias = False)
        self.W_o = nn.Linear(d_out, d_out)
        self.adj = DynamicAdjacency(N, n_heads, d_in, d_regime, dropout = dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, regime_ctx = None):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        edge_bias = self.adj(x, regime_ctx)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale + edge_bias
        attn = self.drop(F.softmax(scores, dim = -1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)


class TemporalCrossAssetAttention(nn.Module):

    def __init__(self, d_in, d_out, n_heads = 4, dropout = 0.1):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.scale = self.d_head ** -0.5
        self.W_q = nn.Linear(d_in, d_out, bias = False)
        self.W_k = nn.Linear(d_in, d_out, bias = False)
        self.W_v = nn.Linear(d_in, d_out, bias = False)
        self.W_o = nn.Linear(d_out, d_out)
        self.drop = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.randn(8, d_in) * 0.02)

    def forward(self, recent):
        B, N, T, D = recent.shape
        T_eff = min(T, self.pos_embed.shape[0])
        pos = self.pos_embed[-T_eff:].unsqueeze(0).unsqueeze(0)
        x = recent + pos
        q_src = x[:, :, -1, :]
        k_src = x.reshape(B, N * T_eff, D)
        Q = self.W_q(q_src).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(k_src).view(B, N * T_eff, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(k_src).view(B, N * T_eff, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = self.drop(F.softmax(scores, dim = -1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)


class RegimeFiLM(nn.Module):

    def __init__(self, d_ctx, d_feat):
        super().__init__()
        self.proj = nn.Linear(d_ctx, d_ctx * 2)
        self.to_scale = nn.Linear(d_ctx * 2, d_feat)
        self.to_shift = nn.Linear(d_ctx * 2, d_feat)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, ctx):
        h = F.gelu(self.proj(ctx))
        s = self.to_scale(h).unsqueeze(1)
        b = self.to_shift(h).unsqueeze(1)
        return x * (1.0 + s) + b



class ReturnPredictionHead(nn.Module):

    def __init__(self, d_enc, d_ctx, d_hidden = 48, dropout = 0.15):
        super().__init__()
        d_in = d_enc + d_ctx
        self.return_net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1))
        self.logvar_net = nn.Sequential(
            nn.Linear(d_in, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1))
        nn.init.zeros_(self.return_net[-1].weight)
        nn.init.zeros_(self.return_net[-1].bias)
        nn.init.constant_(self.logvar_net[-1].bias, -2.0)

    def forward(self, h_per_asset, regime_ctx):
        B, N, D = h_per_asset.shape
        ctx = regime_ctx.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([h_per_asset, ctx], dim = -1)
        flat = inp.reshape(B * N, -1)
        pred_ret = self.return_net(flat).view(B, N)
        log_var = self.logvar_net(flat).view(B, N)
        return pred_ret, log_var


class PortfolioConstructor:

    def __init__(self, cost_rate = 0.0015, gate_sensitivity = 12.0):
        self.cost_rate = cost_rate
        self.gate_sensitivity = gate_sensitivity

    def construct(self, pred_returns, log_var, regime_ctx, prev_weights = None):
        direction_prob = regime_ctx[:, 0:1]
        dir_confidence = regime_ctx[:, 2:3]
        transition_intensity = regime_ctx[:, 4:5]
        bull_signal = (0.5 - direction_prob) * dir_confidence * (1.0 - transition_intensity * 0.5)
        gate = torch.sigmoid(bull_signal * self.gate_sensitivity)
        precision = (-log_var).exp().clamp(max = 100.0)
        alpha = pred_returns * precision
        pos_alpha = F.relu(alpha)
        denom = pos_alpha.sum(dim = -1, keepdim = True) + 1e-8
        raw_w = pos_alpha / denom
        target = raw_w * gate
        if prev_weights is not None:
            delta = target - prev_weights
            expected_gain = (delta * pred_returns).abs()
            cost = delta.abs() * self.cost_rate
            skip = (expected_gain < cost).float()
            target = prev_weights + delta * (1.0 - skip)
        return target, gate.squeeze(-1)


class Model2(nn.Module):

    def __init__(self, F_4h, F_15m, F_1h, D_time_4h, D_time_15m, D_time_1h,
                 N_assets = 20, d_regime = 5, d_model = 32, d_lstm = 32, d_cross = 48,
                 n_cross_heads = 4, dropout = 0.12, embed_drop = 0.5, t_recent = 4):
        super().__init__()
        self.N = N_assets
        self.embed_drop = embed_drop
        self.t_recent = t_recent
        self.enc_4h = AssetTemporalEncoder(F_4h, D_time_4h, d_model = d_model, d_lstm = d_lstm, dropout = dropout)
        self.enc_15m = SubHourlyEncoder(F_15m, D_time_15m, d_model, dropout = dropout)
        self.enc_1h = SubHourlyEncoder(F_1h, D_time_1h, d_model, dropout = dropout)
        self.temporal_xa = TemporalCrossAssetAttention(d_lstm, d_lstm, n_heads = n_cross_heads, dropout = dropout)
        self.temporal_xa_norm = nn.LayerNorm(d_lstm)
        d_fuse = d_lstm + d_model * 2
        self.asset_fuse = nn.Sequential(
            nn.Linear(d_fuse, d_cross),
            nn.GELU(),
            nn.LayerNorm(d_cross),
            nn.Dropout(dropout))
        self.asset_embed = nn.Parameter(torch.randn(N_assets, d_cross) * 0.10)
        self.film = RegimeFiLM(d_regime, d_cross)
        self.cross_attn = CrossAssetAttention(d_cross, d_cross, N_assets, n_heads = n_cross_heads, d_regime = d_regime, dropout = dropout)
        self.attn_norm = nn.LayerNorm(d_cross)
        self.ffn = GRN(d_cross, d_cross, dropout = dropout)
        self.ffn_norm = nn.LayerNorm(d_cross)
        self.pred_head = ReturnPredictionHead(d_enc = d_cross, d_ctx = d_regime, dropout = dropout)

    def forward(self, f4h, te4h, f15m, te15m, f1h, te1h, m1_out):
        B = f4h.shape[0]
        N = self.N
        h_4h, recent_4h = self.enc_4h(f4h, te4h, t_recent = self.t_recent)
        h_15m = self.enc_15m(f15m, te15m)
        h_1h = self.enc_1h(f1h, te1h)
        h_4h = self.temporal_xa_norm(h_4h + self.temporal_xa(recent_4h))
        h = self.asset_fuse(torch.cat([h_4h, h_15m, h_1h], dim = -1))
        if self.training and self.embed_drop > 0:
            keep = 1.0 - self.embed_drop
            mask = torch.bernoulli(torch.full((N,), keep, device = h.device))
            h = h + self.asset_embed[None, :, :] * mask.unsqueeze(0).unsqueeze(-1) / keep
        else:
            h = h + self.asset_embed[None, :, :]
        h = self.film(h, m1_out)
        h = self.attn_norm(h + self.cross_attn(h, regime_ctx = m1_out))
        h = self.ffn_norm(h + self.ffn(h))
        pred_ret, log_var = self.pred_head(h, m1_out)
        return {"pred_ret": pred_ret, "log_var": log_var}
