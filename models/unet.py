import torch
import torch.nn as nn
import torch.nn.functional as F

from models.semantic_encoder import SemanticEncoder


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, t):
        emb = t.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, cond_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_mlp = nn.Linear(t_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, out_channels),
                nn.SiLU(),
                nn.Linear(out_channels, out_channels)
            )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t, cond=None):
        h = self.block1(x)
        h += self.time_mlp(t)[..., None, None]

        if cond is not None and hasattr(self, 'cond_mlp'):
            cond_feat = self.cond_mlp(cond)
            h += cond_feat[..., None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads

        self.context_proj = nn.Linear(context_dim, query_dim) if context_dim != query_dim else nn.Identity()
        self.to_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(query_dim, 2 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, query_dim)

    def forward(self, x, context):
        if context is None:
            return x

        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)

        context = self.context_proj(context)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: t.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2),
            (q, k, v)
        )

        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5),
            dim=-1
        )
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, -1)
        return self.to_out(out).permute(0, 2, 1).reshape(b, -1, h, w)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=64, dim_mults=(1, 2, 4, 8),
                 cond_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        dims = [base_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Conditioning
        if cond_dim is not None:
            self.semantic_encoder = SemanticEncoder(in_channels + 1, cond_dim)
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Downsample blocks
        self.downs = nn.ModuleList()
        curr_dim = base_dim
        for dim_out in dims[1:]:
            self.downs.append(nn.ModuleList([
                ResidualBlock(curr_dim, curr_dim, time_dim, time_dim),
                ResidualBlock(curr_dim, curr_dim, time_dim, time_dim),
                CrossAttention(query_dim=curr_dim, context_dim=time_dim),
                nn.Conv2d(curr_dim, dim_out, 3, stride=2, padding=1)
            ]))
            curr_dim = dim_out

        # Middle blocks
        self.mid_block1 = ResidualBlock(curr_dim, curr_dim, time_dim, time_dim)
        self.mid_attn = CrossAttention(query_dim=curr_dim, context_dim=time_dim) if cond_dim else nn.Identity()
        self.mid_block2 = ResidualBlock(curr_dim, curr_dim, time_dim, time_dim)

        # Upsample blocks
        self.ups = nn.ModuleList()
        up_in_out = [(d2, d1) for d1, d2 in reversed(in_out)]
        for dim_in, dim_out in up_in_out:
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                ResidualBlock(dim_out * 2, dim_out, time_dim, time_dim),
                ResidualBlock(dim_out, dim_out, time_dim, time_dim),
                CrossAttention(query_dim=dim_out, context_dim=time_dim) if cond_dim else nn.Identity(),
            ]))

        # Final layers
        self.final_res_block = ResidualBlock(base_dim * 2, base_dim, time_dim)
        self.final_conv = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x, t, cond=None):
        t_emb = self.time_mlp(t)

        # Process conditioning
        processed_cond = torch.zeros((x.shape[0], 512)).cuda()
        # if self.cond_dim is not None and cond is not None:
        #     encoded_cond = self.semantic_encoder(cond)
        #     processed_cond = self.cond_proj(encoded_cond)
        #     processed_cond = torch.clamp(processed_cond, 0, 1)
        #
        #     # Normalised cond
        #     processed_cond = F.normalize(processed_cond, p=2, dim=1)

        x = self.init_conv(x)
        x = torch.clamp(x, 0, 1)
        h = [x]

        # Downsample path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_emb, processed_cond)
            x = block2(x, t_emb, processed_cond)
            x = attn(x, processed_cond) if (self.cond_dim and processed_cond is not None) else x
            h.append(x)
            x = downsample(x)

        # Middle blocks
        x = self.mid_block1(x, t_emb, processed_cond)
        x = self.mid_attn(x, processed_cond) if (self.cond_dim and processed_cond is not None) else x
        x = self.mid_block2(x, t_emb, processed_cond)

        # Upsample path
        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t_emb, processed_cond)
            x = block2(x, t_emb, processed_cond)
            x = attn(x, processed_cond) if (self.cond_dim and processed_cond is not None) else x

        # Final convolution
        x = torch.cat([x, h.pop()], dim=1)
        x = self.final_res_block(x, t_emb)

        return self.final_conv(x)
