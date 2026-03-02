from torch.utils.data import Dataset
import numpy as np

class BuildDataset(Dataset):
    def __init__(
            self,
            data: np.ndarray,       # (T, C) 스케일된 시계열 값
            data_ts: np.ndarray,   # (T, F) 시간 피처
            seq_len: int,           # encoder 입력 길이
            label_len: int,         # decoder 시작 토큰 길이 (encoder와 겹치는 부분. TimesNet은 decoder가 없으므로 lable_len = 0)
            pred_len: int
            ):
        self.data = data
        self.data_ts = data_ts
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # 유효한 슬라이딩 윈도우 수 확인
        self.valid_window = len(data) - seq_len - pred_len + 1
        assert self.valid_window > 0, (
            f"데이터가 너무 짧습니다: T= {len(data)}, seq_len= {seq_len}, pred_len= {pred_len}"
        )
        return None
    
    def __getitem__(self, idx):
        # encoder 구간
        enc_start = idx
        enc_end = idx + self.seq_len

        # decoder 구간 (label_len만큼 encoder와 겹치게 설정)
        dec_start = enc_end - self.label_len
        dec_end = enc_end + self.pred_len

        seq_x = self.data[enc_start : enc_end]
        seq_x_mark = self.data_ts[enc_start : enc_end]

        seq_y = self.data[dec_start : dec_end]
        seq_y_mark = self.data_ts[dec_start : dec_end]

        return seq_x, seq_x_mark, seq_y, seq_y_mark
    
    def __len__(self):
        return self.valid_window
