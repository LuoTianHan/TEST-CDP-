#!/bin/bash

# TEST-CDP ETTm1数据集长期预测实验脚本
# 使用方法: bash scripts/run_ettm1.sh

echo "============================================"
echo "TEST-CDP: ETTm1 长期预测实验"
echo "============================================"

# 模型和数据配置
MODEL="TEST_CDP_Llama"
DATA="ETTm1"
ROOT_PATH="./data/ETT-small/"
DATA_PATH="ETTm1.csv"
FEATURES="M"
ENC_IN=7

# 序列长度配置
SEQ_LEN=672
LABEL_LEN=576
TOKEN_LEN=96

# 预测长度列表
PRED_LENS=(96 192 336 720)

# 遍历所有预测长度
for PRED_LEN in "${PRED_LENS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "预测长度: $PRED_LEN"
    echo "--------------------------------------------"
    
    python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "${DATA}_${SEQ_LEN}_${PRED_LEN}" \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --features $FEATURES \
        --target OT \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --token_len $TOKEN_LEN \
        --enc_in $ENC_IN \
        --dec_in $ENC_IN \
        --c_out $ENC_IN \
        --hidden_dim 512 \
        --llm_embed_dim 4096 \
        --prompt_length 16 \
        --num_text_prototypes 10 \
        --num_gat_layers 2 \
        --num_heads 4 \
        --tau_pos 0.5 \
        --tau_neg 0.1 \
        --cpcl_weight 0.1 \
        --dpg_beta 0.3 \
        --dropout 0.1 \
        --train_epochs 20 \
        --batch_size 32 \
        --patience 3 \
        --learning_rate 0.0001 \
        --weight_decay 0.01 \
        --lradj cosine \
        --use_amp \
        --gpu 0 \
        --seed 2021 \
        --des "TEST_CDP_${DATA}_${PRED_LEN}"
    
    echo "预测长度 $PRED_LEN 实验完成"
done

echo ""
echo "============================================"
echo "ETTm1 所有实验完成!"
echo "============================================"
