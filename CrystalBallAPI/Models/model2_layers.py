import torch
import torch.nn as nn
import torch.nn.functional as F
from model1_layers import GRN, CausalConv1d


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


class CrossAssetAttention(nn.Module):
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

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
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
        self.to_scale = nn.Linear(d_ctx, d_feat)
        self.to_shift = nn.Linear(d_ctx, d_feat)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, ctx):
        s = self.to_scale(ctx).unsqueeze(1)
        b = self.to_shift(ctx).unsqueeze(1)
        return x * (1.0 + s) + b


class DavisNormanHead(nn.Module):
    def __init__(self, d_enc, F_char, d_ctx, d_target = 48, d_gate = 24, dropout = 0.15,
                 band_init = 0.05, band_min = 0.005, band_max = 0.30, sharpness = 30.0):
        super().__init__()
        self.target_net = nn.Sequential(
            nn.Linear(d_enc + F_char, d_target),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_target, d_target // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_target // 2, 1))
        self.band_net = nn.Sequential(
            nn.Linear(d_enc + F_char, d_target // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_target // 2, 1))
        self.gate_net = nn.Sequential(
            nn.Linear(d_enc + d_ctx, d_gate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_gate, 1))
        self.log_temp = nn.Parameter(torch.tensor(0.5).log())
        self.register_buffer("inv_vol_prior", torch.zeros(1))
        self.band_min = band_min
        self.band_max = band_max
        self.band_init = band_init
        self.sharpness = sharpness

    def set_inv_vol_prior(self, asset_vol):
        inv_vol = 1.0 / asset_vol.clamp(min = 1e-6)
        log_prior = inv_vol.log()
        prior = (log_prior - log_prior.mean()).to(self.inv_vol_prior.device)
        self.inv_vol_prior = prior.detach()

    def forward(self, h_per_asset, chars, m1_out):
        B, N, D = h_per_asset.shape
        ta_in = torch.cat([h_per_asset, chars], dim = -1)
        ta_flat = ta_in.reshape(B * N, -1)
        a = self.target_net(ta_flat).squeeze(-1).view(B, N)
        a = a + self.inv_vol_prior
        temp = self.log_temp.exp().clamp(min = 0.25, max = 8.0)
        target = F.softmax(a / temp, dim = -1)
        band_raw = self.band_net(ta_flat).squeeze(-1).view(B, N)
        band = self.band_min + (self.band_max - self.band_min) * torch.sigmoid(band_raw)
        pooled = h_per_asset.mean(dim = 1)
        ctx_in = torch.cat([pooled, m1_out], dim = -1)
        gate = torch.sigmoid(self.gate_net(ctx_in)).squeeze(-1)
        return target, gate, band, a


class Model2(nn.Module):
    def __init__(self,
                 F_1h, F_15m, F_30m,
                 D_time_1h, D_time_15m, D_time_30m,
                 N_assets = 20, d_regime = 6,
                 d_model = 32, d_lstm = 32, d_cross = 48,
                 n_cross_heads = 4, dropout = 0.12,
                 embed_drop = 0.5, band_sharpness = 30.0,
                 t_recent = 4):
        super().__init__()
        self.N = N_assets
        self.embed_drop = embed_drop
        self.band_sharpness = band_sharpness
        self.t_recent = t_recent
        self.enc_1h = AssetTemporalEncoder(F_1h, D_time_1h, d_model = d_model, d_lstm = d_lstm, dropout = dropout)
        self.enc_15m = SubHourlyEncoder(F_15m, D_time_15m, d_model, dropout = dropout)
        self.enc_30m = SubHourlyEncoder(F_30m, D_time_30m, d_model, dropout = dropout)
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
        self.cross_attn = CrossAssetAttention(d_cross, d_cross, n_heads = n_cross_heads, dropout = dropout)
        self.attn_norm = nn.LayerNorm(d_cross)
        self.ffn = GRN(d_cross, d_cross, dropout = dropout)
        self.ffn_norm = nn.LayerNorm(d_cross)
        self.alloc = DavisNormanHead(d_enc = d_cross, F_char = F_1h, d_ctx = d_regime, dropout = dropout)

    def forward(self, f1h, te1h, f15m, te15m, f30m, te30m, m1_out):
        B = f1h.shape[0]
        N = self.N
        h_1h, recent_1h = self.enc_1h(f1h, te1h, t_recent = self.t_recent)
        h_15m = self.enc_15m(f15m, te15m)
        h_30m = self.enc_30m(f30m, te30m)
        h_1h = self.temporal_xa_norm(h_1h + self.temporal_xa(recent_1h))
        h = self.asset_fuse(torch.cat([h_1h, h_15m, h_30m], dim = -1))
        if self.training and self.embed_drop > 0:
            keep = 1.0 - self.embed_drop
            mask = torch.bernoulli(torch.full((N,), keep, device = h.device))
            h = h + self.asset_embed[None, :, :] * mask.unsqueeze(0).unsqueeze(-1) / keep
        else:
            h = h + self.asset_embed[None, :, :]
        h = self.film(h, m1_out)
        h = self.attn_norm(h + self.cross_attn(h))
        h = self.ffn_norm(h + self.ffn(h))
        chars = f1h[:, -1, :, :]
        target, gate, band, logits = self.alloc(h, chars, m1_out)
        return {"target": target, "gate": gate, "band": band, "logits": logits}
