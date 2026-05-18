#!/bin/bash

# TEST-CDP 全数据集长期预测实验脚本
# 在8个真实数据集上运行实验
# 使用方法: bash scripts/run_all_datasets.sh

echo "============================================"
echo "TEST-CDP: 全数据集长期预测实验"
echo "============================================"

# 通用配置
MODEL="TEST_CDP_Llama"
FEATURES="M"
HIDDEN_DIM=512
LLM_EMBED_DIM=4096
PROMPT_LENGTH=16
NUM_PROTOTYPES=10
NUM_GAT_LAYERS=2
NUM_HEADS=4
DROPOUT=0.1
TRAIN_EPOCHS=20
BATCH_SIZE=32
PATIENCE=3
LR=0.0001
WEIGHT_DECAY=0.01

# 数据集配置数组
# 格式: "数据名称 数据路径 输入变量数 序列长度 数据目录"
DATASETS=(
    "ETTh1 ETTh1.csv 7 672 ./data/ETT-small/"
    "ETTh2 ETTh2.csv 7 672 ./data/ETT-small/"
    "ETTm1 ETTm1.csv 7 672 ./data/ETT-small/"
    "ETTm2 ETTm2.csv 7 672 ./data/ETT-small/"
    "Weather weather.csv 21 672 ./data/weather/"
    "ECL electricity.csv 321 672 ./data/electricity/"
    "Traffic traffic.csv 862 672 ./data/traffic/"
    "ILI national_illness.csv 7 104 ./data/illness/"
)

# 预测长度配置
PRED_LENS_ETT="96 192 336 720"
PRED_LENS_ILI="24 36 48 60"

# 遍历所有数据集
for dataset_config in "${DATASETS[@]}"; do
    # 解析数据集配置
    read -r DATA DATA_PATH ENC_IN SEQ_LEN ROOT_PATH <<< "$dataset_config"
    
    echo ""
    echo "============================================"
    echo "数据集: $DATA"
    echo "============================================"
    
    # 选择预测长度
    if [ "$DATA" = "ILI" ]; then
        PRED_LENS=$PRED_LENS_ILI
        LABEL_LEN=48
    else
        PRED_LENS=$PRED_LENS_ETT
        LABEL_LEN=576
    fi
    
    # 遍历所有预测长度
    for PRED_LEN in $PRED_LENS; do
        echo ""
        echo "--------------------------------------------"
        echo "数据集: $DATA | 预测长度: $PRED_LEN"
        echo "--------------------------------------------"
        
        python run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --model_id "${DATA}_${SEQ_LEN}_${PRED_LEN}" \
            --model $MODEL \
            --data $DATA \
            --root_path "$ROOT_PATH" \
            --data_path "$DATA_PATH" \
            --features $FEATURES \
            --target OT \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --token_len 96 \
            --enc_in $ENC_IN \
            --dec_in $ENC_IN \
            --c_out $ENC_IN \
            --hidden_dim $HIDDEN_DIM \
            --llm_embed_dim $LLM_EMBED_DIM \
            --prompt_length $PROMPT_LENGTH \
            --num_text_prototypes $NUM_PROTOTYPES \
            --num_gat_layers $NUM_GAT_LAYERS \
            --num_heads $NUM_HEADS \
            --tau_pos 0.5 \
            --tau_neg 0.1 \
            --cpcl_weight 0.1 \
            --dpg_beta 0.3 \
            --dropout $DROPOUT \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --patience $PATIENCE \
            --learning_rate $LR \
            --weight_decay $WEIGHT_DECAY \
            --lradj cosine \
            --use_amp \
            --gpu 0 \
            --seed 2021 \
            --des "TEST_CDP_${DATA}_${PRED_LEN}" \
            --visualize
        
        echo "数据集 $DATA 预测长度 $PRED_LEN 完成"
    done
    
    echo "数据集 $DATA 所有预测长度实验完成"
done

echo ""
echo "============================================"
echo "所有数据集实验完成!"
echo "============================================"
