window_size=96
data_path=../dataset/ETT-small/ETTm2.csv # 이거 수정함
data_name=ETTm2
model_name=TimesNet
batch_size=16 # 이거 수정함
# DATAINFO -> DATASET 으로 전부 수정함

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    DATASET.pred_len 96 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_96 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 16

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    DATASET.pred_len 192 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_192 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 16

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    DATASET.pred_len 336 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_336 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 16

accelerate launch main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.window_size $window_size \
    DATASET.datadir $data_path \
    DATASET.dataname $data_name \
    DATASET.pred_len 720 \
    DEFAULT.exp_name forecasting_${data_name}_${window_size}_720 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 16