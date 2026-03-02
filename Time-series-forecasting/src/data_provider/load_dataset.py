from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features_from_date

def load_dataset(
    datadir: str,
    dataname: str,
    split_rate: list,
    seq_len: int,
    time_embedding: list = [True, 'h'], 
    del_feature: list = None
):
    # 1. CSV 파일 찾기
    if os.path.isfile(datadir):
        filepath = datadir
    else:
        filepath = None
        for root, _, files in os.walk(datadir):
            for f in files:
                if dataname in f and f.endswith('.csv'):
                    filepath = os.path.join(root, f)
                    break
                if filepath:
                    break
            if filepath is None:
                raise FileNotFoundError(
                    f"'{dataname}' CSV를 '{datadir}'에서 찾을 수 없습니다."
                )
    # 2. CSV 로드 & 불필요한 피처 제거
    df = pd.read_csv(filepath)

    if del_feature:
        df = df.drop(columns=[c for c in del_feature if c in df.columns])
    
    # 3. 날짜 컬럼 분리 (-> timestamp)
    date_col = df.columns[0]
    date_series = df[date_col]
    data_df = df.drop(columns=[date_col])

    data_arr = data_df.values.astype(np.float32) # (T, C)
    T = len(data_arr)

    # 4. 시간 피처 생성
    timeenc, freq = int(time_embedding[0]), str(time_embedding[1])
    ts_df = time_features_from_date(date_series, timeenc=timeenc, freq=freq)
    ts_arr = ts_df.values.astype(np.float32) #(T, n_time_feats)

    # 5. Train / Val / Test 분할
    ETT_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']

    if dataname in ETT_datasets:
        freq_mult = 4 if 'm' in dataname else 1
        n_train = 12 * 30 * 24 * freq_mult
        n_val = 4 * 30 * 24 * freq_mult
        n_test = 4 * 30 * 24 * freq_mult
    else:
        train_r, val_r, _ = split_rate
        n_train = int(T*train_r)
        n_val = int(T*val_r)
        n_test = T - n_train - n_val
    
    # val/test는 seq_len만큼 앞으로 당겨서 시작 -> val/test 첫 샘플의 look-back 확보
    trn = data_arr[:n_train]
    trn_ts = ts_arr[:n_train]
    val = data_arr[n_train - seq_len : n_train + n_val]
    val_ts = ts_arr[n_train - seq_len : n_train + n_val]
    tst = data_arr[n_train + n_val - seq_len :]
    tst_ts = ts_arr[n_train + n_val - seq_len :]
    var = data_arr.shape[1]
    return trn, trn_ts, val, val_ts, tst, tst_ts, var