from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(trn_raw, dev_raw, tst_raw, scaler_type = 'standard'):
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'minmax_m1p1':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif scaler_type == 'minmax_square':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError(f"Unknown scaler_type: '{scaler_type}'."
                         "Choose from: 'standard', 'minmax', 'minmax_m1p1', 'minmax_square'")
    
    scaler.fit(trn_raw)

    if scaler_type == 'minmax_square':
        trn = (scaler.transform(trn_raw) ** 2).astype(np.float32)
        dev = (scaler.transform(dev_raw) ** 2).astype(np.float32)
        tst = (scaler.transform(tst_raw) ** 2).astype(np.float32)
    else:
        trn = (scaler.transform(trn_raw)).astype(np.float32)
        dev = (scaler.transform(dev_raw)).astype(np.float32)
        tst = (scaler.transform(tst_raw)).astype(np.float32)

    return trn, dev, tst