import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

def decode_image(vae, latent):
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image


class GRUFrameWork(nn.Module):
    def __init__(self, opts, vae=None, hidden_dim=64):
        super(GRUFrameWork, self).__init__()
        self.opts = opts

        self.depth_encoder = ProjectionInputDepth(hidden_dim=hidden_dim // 2, out_chs=hidden_dim // 2)
        self.context_encoder = ProjectionInputContext(hidden_dim=hidden_dim // 2, out_chs=hidden_dim // 2)

        self.project = nn.Conv2d(3, hidden_dim, 1, padding=0)
        self.d_head = DHead(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.apply(self._init_weights)

        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim)
        self.gru.init_weights()
        self.ss_head = ScaleShiftTransformer(hidden_dim=hidden_dim)  # ScaleShiftHead(input_dim=hidden_dim, hidden_dim=hidden_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, context, gru_hidden, seq_len=2):
        # depth: [B, 1, H, W]
        # context: [B, 4, H/8, W/8]
        # gru_hidden: [B, 3,H,W]
        if self.opts.link_mode == 'mean':
            depth = gru_hidden.mean(dim=1, keepdim=True)
        elif self.opts.link_mode == 'first':
            depth = gru_hidden[:, :1]
        depth = torch.clamp(depth, -1, 1) / 2 + 0.5  # [any,any] -> [0, 1]
        output = {"pred": depth}
        depth_list, depth_feature = [], []
        depth_list.append(depth)
        gru_hidden = torch.tanh(self.project(gru_hidden))  # [B, 64, H, W]
        depth_feature.append(gru_hidden.mean(dim=1, keepdim=True))
        context = self.context_encoder(context)  # B, 64, H, W
        for i in range(seq_len):
            scale, shift = self.ss_head(gru_hidden)
            depth = depth.detach() * scale.view(-1, 1, 1, 1) + shift.view(-1, 1, 1, 1)
            input_features = self.depth_encoder(depth)  # B, 64, H, W
            input_c = torch.cat([input_features, context], dim=1)
            gru_hidden = self.gru(gru_hidden, input_c)
            depth_feature.append(gru_hidden.mean(dim=1, keepdim=True))
            delta_d = self.d_head(gru_hidden)
            depth = (depth + delta_d).clamp(0, 3)
            depth_list.append(depth)
        output["depth_list"], output["depth_feature"] = depth_list, depth_feature
        return output


class ScaleShiftHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.scale_fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 保证输出为正
        )
        self.shift_fc = nn.Linear(hidden_dim, 1)

        nn.init.constant_(self.scale_fc[0].weight, 0)
        nn.init.constant_(self.scale_fc[0].bias, 0.54)  # Softplus(0.54)=~0.999
        nn.init.constant_(self.shift_fc.weight, 0)
        nn.init.constant_(self.shift_fc.bias, 0)

    def forward(self, x):
        feat = self.conv(x).squeeze(-1).squeeze(-1)  # [B, hidden_dim]
        scale = self.scale_fc(feat)  # initially 1.0
        shift = self.shift_fc(feat)  # initially 0.0
        return scale, shift  # [B,1], [B,1]


class ScaleShiftTransformer(nn.Module):

    def __init__(self, hidden_dim=64, num_queries=2):
        super().__init__()
        self.scale_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.shift_query = nn.Parameter(torch.randn(1, hidden_dim))

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self._init_attention()

        self.scale_head = self._build_head(hidden_dim, init_bias=0.541)  # Softplus(0.541)=1.0
        self.shift_head = self._build_head(hidden_dim, init_bias=0.0)

    def _init_queries(self, hidden_dim):
        nn.init.trunc_normal_(self.scale_query, mean=0.541, std=0.02, a=0.4, b=0.7)
        nn.init.trunc_normal_(self.shift_query, mean=0.0, std=0.01)

    def _init_attention(self):
        for p in self.attn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_head(self, hidden_dim, init_bias):
        head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.kaiming_normal_(head[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(head[0].bias, 0.0)
        nn.init.xavier_normal_(head[2].weight, gain=0.01)
        nn.init.constant_(head[2].bias, init_bias)
        return head

    def forward(self, gru_hidden):
        """
        gru_hidden: [B, C, H, W] -> 展平为 [HW, B, C]
        """
        B, C, H, W = gru_hidden.shape
        gru_flatten = gru_hidden.flatten(2).permute(2, 0, 1)  # [HW, B, C]

        queries = torch.stack([self.scale_query, self.shift_query], dim=1)
        queries = queries.expand(B, -1, -1).permute(1, 0, 2)  # [2, B, C]

        attn_out, _ = self.attn(
            query=queries,
            key=gru_flatten,
            value=gru_flatten,
            need_weights=False
        )  # [2, B, C]

        scale_feat = attn_out[0]  # [B, C]
        shift_feat = attn_out[1]  # [B, C]

        scale = self.scale_head(scale_feat).unsqueeze(-1)  # [B, 1]
        shift = self.shift_head(shift_feat).unsqueeze(-1)  # [B, 1]

        return scale, shift


class ProjectionInputContext(nn.Module):
    def __init__(self, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(4, hidden_dim, 7, padding=3)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convc3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convc4 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)

    def forward(self, context):
        c = F.relu(self.convc1(context))
        c = upsample(c, scale_factor=2)
        c = F.relu(self.convc2(c))
        c = upsample(c, scale_factor=2)
        c = F.relu(self.convc3(c))
        c = upsample(c, scale_factor=2)
        c = F.relu(self.convc4(c))
        return c


class ProjectionInputDepth(nn.Module):
    def __init__(self, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd4 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)

    def forward(self, depth):
        d = F.relu(self.convd1(depth))
        d = F.relu(self.convd2(d))
        d = F.relu(self.convd3(d))
        d = F.relu(self.convd4(d))

        return d


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=128):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class DHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(DHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_du, act_fn=F.tanh):
        out = self.conv2(self.relu(self.conv1(x_du)))
        return act_fn(out)


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def downsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Downsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=1 / scale_factor, mode=mode, align_corners=align_corners)
