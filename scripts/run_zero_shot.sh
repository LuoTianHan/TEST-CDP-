#!/bin/bash

# TEST-CDP 零样本跨域迁移实验脚本
# 测试模型在未见过的数据集上的泛化能力
# 使用方法: bash scripts/run_zero_shot.sh

echo "============================================"
echo "TEST-CDP: 零样本跨域预测实验"
echo "============================================"

MODEL="TEST_CDP_Llama"
SEQ_LEN=672
LABEL_LEN=576
PRED_LEN=96

# 跨域迁移任务列表
# 格式: "源数据集 目标数据集 数据路径"
TRANSFER_TASKS=(
    "ETTh1 ETTh2 ./data/ETT-small/ ETTh1.csv ETTh2.csv 7"
    "ETTh1 ETTm2 ./data/ETT-small/ ETTh1.csv ETTm2.csv 7"
    "ETTh2 ETTh1 ./data/ETT-small/ ETTh2.csv ETTh1.csv 7"
    "ETTh2 ETTm2 ./data/ETT-small/ ETTh2.csv ETTm2.csv 7"
    "ETTm1 ETTh2 ./data/ETT-small/ ETTm1.csv ETTh2.csv 7"
    "ETTm1 ETTm2 ./data/ETT-small/ ETTm1.csv ETTm2.csv 7"
    "ETTm2 ETTh2 ./data/ETT-small/ ETTm2.csv ETTh2.csv 7"
    "ETTm2 ETTm1 ./data/ETT-small/ ETTm2.csv ETTm1.csv 7"
)

for task_config in "${TRANSFER_TASKS[@]}"; do
    read -r SOURCE TARGET ROOT_PATH SOURCE_PATH TARGET_PATH ENC_IN <<< "$task_config"
    
    echo ""
    echo "--------------------------------------------"
    echo "跨域任务: $SOURCE -> $TARGET"
    echo "--------------------------------------------"
    
    # 训练阶段：在源数据集上训练
    echo "[阶段1/2] 在源数据集 $SOURCE 上训练..."
    python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "${SOURCE}_to_${TARGET}_pretrain" \
        --model $MODEL \
        --data $SOURCE \
        --root_path "$ROOT_PATH" \
        --data_path "$SOURCE_PATH" \
        --features M \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --enc_in $ENC_IN \
        --hidden_dim 512 \
        --llm_embed_dim 4096 \
        --prompt_length 16 \
        --train_epochs 10 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --lradj cosine \
        --gpu 0 \
        --des "TEST_CDP_zero_shot_pretrain"
    
    # 测试阶段：在目标数据集上零样本测试
    echo "[阶段2/2] 在目标数据集 $TARGET 上零样本测试..."
    python run.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --model_id "${SOURCE}_to_${TARGET}_test" \
        --model $MODEL \
        --data $TARGET \
        --root_path "$ROOT_PATH" \
        --data_path "$TARGET_PATH" \
        --features M \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --enc_in $ENC_IN \
        --hidden_dim 512 \
        --llm_embed_dim 4096 \
        --gpu 0 \
        --des "TEST_CDP_zero_shot_eval"
    
    echo "跨域任务 $SOURCE -> $TARGET 完成"
done

echo ""
echo "============================================"
echo "零样本跨域实验全部完成!"
echo "============================================"
