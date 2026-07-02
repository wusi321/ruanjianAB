#!/bin/bash
# batch_test_score.sh
# 批量测试不同 SCORE_THRESH 对 L 模型的影响
# 用法: bash batch_test_score.sh
# 需要先准备好 data.txt（验证集图片列表）

PREDICT_PY="/home/aistudio/submission/predict.py"
DATA_TXT="/home/aistudio/submission/data.txt"
RESULT_DIR="/home/aistudio/score_test_results"
EVAL_SCRIPT="/home/aistudio/work/eval_firebig.py"

THRESHOLDS=(0.05 0.10 0.15 0.20 0.25 0.28 0.30)

mkdir -p $RESULT_DIR
echo "SCORE_THRESH,F1,TP,FP,FN,FPS" > $RESULT_DIR/scores.csv

for SCORE in "${THRESHOLDS[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing SCORE_THRESH = $SCORE"
    echo "========================================"

    # 修改 predict.py 中的 SCORE_THRESH
    sed -i "s/^SCORE_THRESH = .*/SCORE_THRESH = $SCORE/" $PREDICT_PY

    # 跑预测
    RESULT_JSON="$RESULT_DIR/result_$SCORE.json"
    python $PREDICT_PY $DATA_TXT $RESULT_JSON

    # 跑评估
    python $EVAL_SCRIPT >> $RESULT_DIR/eval_$SCORE.log

    # 提取关键指标
    F1=$(grep "F1 Score" $RESULT_DIR/eval_$SCORE.log | awk "{print \$NF}")
    TP=$(grep "TP" $RESULT_DIR/eval_$SCORE.log | awk "{print \$NF}")
    FP=$(grep "FP" $RESULT_DIR/eval_$SCORE.log | awk "{print \$NF}")
    FN=$(grep "FN" $RESULT_DIR/eval_$SCORE.log | awk "{print \$NF}")
    FPS=$(grep "FPS" $RESULT_DIR/eval_$SCORE.log | awk "{print \$NF}")

    echo "$SCORE,$F1,$TP,$FP,$FN,$FPS" >> $RESULT_DIR/scores.csv
    echo "SCORE=$SCORE  F1=$F1"
done

echo ""
echo "========================================"
echo "所有测试完成！结果汇总:"
echo "========================================"
cat $RESULT_DIR/scores.csv

# 恢复默认值
sed -i "s/^SCORE_THRESH = .*/SCORE_THRESH = 0.10/" $PREDICT_PY
