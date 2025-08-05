# 題目 Q3: 混淆矩陣與效能指標實作
# (a) 撰寫函數輸出 TP, TN, FP, FN
# (b) 計算 accuracy, precision, recall, F1 分數（不可使用 sklearn）

import numpy as np

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    return TP, TN, FP, FN, accuracy, precision, recall, f1

# 測試資料
y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1]

results = compute_metrics(y_true, y_pred)
print("TP, TN, FP, FN:", results[:4])
print("Accuracy, Precision, Recall, F1:", results[4:])