import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding

def FFT_for_Period(x, k= 2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim= 1)  # FFT 수행 xf: (B, T//2+1, C) <- 복소수 주파수 성분
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1) # 진폭의 평균 내어 A 계산
    frequency_list[0] = 0 # 주파수가 0인 DC 성분 제거
    _, top_list = torch.topk(frequency_list, k) # A 기준으로 top K 주파수 인덱스 추출
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list # 주기 계산 (T // 주파수 인덱스 = 주기)
    
    return period, abs(xf).mean(-1)[:, top_list]
    # period: (k, _) <- 주기 길이들
    # abs(xf).mean(-1): (B, T//2+1) <- 배치별, 주파수별 진폭
    # abs(xf).mean(-1)[:, top_list]: (B, k) <- 그 중에서 top k 개의 주파수만 선별

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len    = getattr(configs, 'seq_len', 96)
        self.pred_len   = getattr(configs, 'pred_len', 96)
        self.k          = getattr(configs, 'top_k', 5)

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels= configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size() # (batch, seq_len+pred_len, d_model)
        # 1. FFT로 top-k 주기 탐색
        period_list, period_weight = FFT_for_Period(x, self.k)

        # 2. 각 주기마다 1D -> 2D -> Conv -> 1D
        res = []
        for i in range(self.k):
            period = period_list[i]

            # 패딩: (seq_len+pred_len)이 period의 배수가 되도록
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - T, N]).to(x.device)
                out = torch.cat([x, padding], dim = 1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            # 1D -> 2D reshape: (B, N, length//period, period)
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()

            # 2D Conv
            out = self.conv(out)

            # 2D -> 1D reshape: (B, length, N)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # 원래 길이로 자르기
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # 3. FFT 진폭 가중치로 weighted sum
        res = torch.stack(res, dim = -1)  # (B, T, N, k)

        period_weight = F.softmax(period_weight, dim = 1)       # (B, k)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1) # (B, 1, 1, k)
        period_weight = period_weight.repeat(1, T, N, 1)

        res = torch.sum(res * period_weight, dim = -1) # (B, T, N)
        # residual connection
        res = res + x
        return res

class TimesNet(nn.Module):

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)
        self.label_len = getattr(configs, 'label_len', 0)

        # TimesBlock e_layers개 스택
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Forecasting head
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias = True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (B, seq_len, d_model)

        # 시간축 확장: seq_len -> seq_len + pred_len
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesBlock 스택
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out) # (B, seq_len + pred_len, c_out)

        # Denormalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 마지막 pred_len 스텝만 반환
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :] # (B, pred_len, c_out)