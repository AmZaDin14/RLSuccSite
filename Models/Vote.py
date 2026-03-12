import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, recall_score, average_precision_score
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import OneHotCategorical
from torchrl.data import DiscreteTensorSpec
from torchrl.modules import ProbabilisticActor
import torch.nn.functional as F
from Features import GetProtT5_K_4, GetTPEMPPS_CCP
import pandas as pd

device = "cpu"

ProtT5_test_negative = BASE_DIR / "Dataset/test/features/test_negative_ProtT5_features_K.csv"
ProtT5_test_positive = BASE_DIR / "Dataset/test/features/test_positive_ProtT5_features_K.csv"
ProtT5_train_negative = BASE_DIR / "Dataset/train/features/train_negative_ProtT5_features_K.csv"
ProtT5_train_positive = BASE_DIR / "Dataset/train/features/train_positive_ProtT5_features_K.csv"

ZccFCCP_test_negative = BASE_DIR / "Dataset/test/fasta/test_negative_sites.fasta"
ZccFCCP_test_positive = BASE_DIR / "Dataset/test/fasta/test_positive_sites.fasta"
ZccFCCP_train_negative = BASE_DIR / "Dataset/train/fasta/train_negative_sites.fasta"
ZccFCCP_train_positive = BASE_DIR / "Dataset/train/fasta/train_positive_sites.fasta"
# -----------------------------------------------------ProtT5模型-----------------------------------------------------
X_ProtT5_train, y_train, X_ProtT5_test, y_test, r = GetProtT5_K_4(ProtT5_train_negative, ProtT5_train_positive,
                                                                  ProtT5_test_negative, ProtT5_test_positive)
X_ProtT5_train = X_ProtT5_train.astype(np.float32)
X_ProtT5_test = X_ProtT5_test.astype(np.float32)
actor_net_ProtT5 = nn.Sequential(
    nn.LazyLinear(X_ProtT5_train.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)

policy_module_ProtT5 = TensorDictModule(
    actor_net_ProtT5, in_keys=["observation"], out_keys=["logits"]
)

policy_module_ProtT5 = ProbabilisticActor(
    module=policy_module_ProtT5,
    spec=DiscreteTensorSpec(2),
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)

model_path_ProtT5 = 'ProtT5_N10_ACC7142_MCC3513_SN7191_SP7130.pth'
policy_module_ProtT5.load_state_dict(torch.load(model_path_ProtT5, map_location=device))
policy_module_ProtT5.eval()
# -----------------------------------------------------ZccFCCP模型-----------------------------------------------------
X_ZccFCCP_train, y_train, X_ZccFCCP_test, y_test, r = GetTPEMPPS_CCP(ZccFCCP_train_negative, ZccFCCP_train_positive,
                                                                     ZccFCCP_test_negative, ZccFCCP_test_positive)
X_ZccFCCP_train = X_ZccFCCP_train.astype(np.float32)
X_ZccFCCP_test = X_ZccFCCP_test.astype(np.float32)
actor_net_ZccFCCP = nn.Sequential(
    nn.LazyLinear(X_ZccFCCP_train.shape[1], device=device),
    nn.ReLU(),
    nn.LazyLinear(2, device=device),
)

policy_module_ZccFCCP = TensorDictModule(
    actor_net_ZccFCCP, in_keys=["observation"], out_keys=["logits"]
)

policy_module_ZccFCCP = ProbabilisticActor(
    module=policy_module_ZccFCCP,
    spec=DiscreteTensorSpec(2),
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)

model_path_ZccFCCP = 'TPEMPPS_CCP_ACC7083_MCC3307_SN6943_SP7116.pth'
policy_module_ZccFCCP.load_state_dict(torch.load(model_path_ZccFCCP, map_location=device))
policy_module_ZccFCCP.eval()

# -------------------------投票预测----------------------------------------------------
# 存储预测结果
predicted_actions = []
Probabilities = []
with torch.no_grad():
    weight1 = 0.5
    weight2 = 0.5
    for i in range(X_ProtT5_test.shape[0]):
        td1 = TensorDict(
            {},
            [],
        )
        td1["observation"] = X_ProtT5_test[i]
        # 使用加载的模型进行预测
        logits1 = policy_module_ProtT5(td1)["logits"]

        td2 = TensorDict(
            {},
            [],
        )
        td2["observation"] = X_ZccFCCP_test[i]
        # 使用加载的模型进行预测
        logits2 = policy_module_ZccFCCP(td2)["logits"]

        weighted_logits1 = logits1 * weight1
        weighted_logits2 = logits2 * weight2

        # 计算概率的平均值
        avg_logits = (weighted_logits1 + weighted_logits2) / 2
        avg_probs = F.softmax(avg_logits, dim=-1)
        # 概率
        Probabilities.append(avg_probs[1])

        # 获取概率最大的类别作为最终预测结果
        prediction = avg_probs.argmax(dim=-1)
        predicted_actions.append(prediction)

TN, FP, FN, TP = confusion_matrix(y_test, predicted_actions, labels=[0, 1]).ravel()
numerator = (TP * TN) - (FP * FN)
denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
MCC = numerator / denominator if denominator != 0 else 0
Acc = (TP + TN) / (TN + FP + FN + TP)
Sn = TP / (TP + FN) if TP + FN else 0
Sp = TN / (TN + FP) if TN + FP else 0
G_mean = np.sqrt(Sn * Sp)
F1 = f1_score(y_test, predicted_actions, zero_division=0)
Recall = recall_score(y_test, predicted_actions)
Precision = TP / (TP + FP) if TP + FP else 0
AUC = roc_auc_score(y_test, Probabilities)
AP = average_precision_score(y_test, Probabilities)
BalancedAccuracy = (recall_score(y_test, predicted_actions) +
                    (confusion_matrix(y_test, predicted_actions).ravel()[0] /
                     (confusion_matrix(y_test, predicted_actions).ravel()[0] +
                      confusion_matrix(y_test, predicted_actions).ravel()[1]))) / 2
print(" ")
print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
print("BAcc:", "{:.4f}".format(BalancedAccuracy), "MCC:", "{:.4f}".format(MCC), "Sn:", "{:.4f}".format(Sn), "Sp:",
      "{:.4f}".format(Sp), "F1:", "{:.4f}".format(F1))

# 计算性能指标
metrics_dict = {
    'Balanced Accuracy': BalancedAccuracy,
    'MCC': MCC,
    'Sensitivity': Sn,
    'Specificity': Sp,
    'F1 Score': F1,
}
# 输出和保存性能指标
df_metrics = pd.DataFrame([metrics_dict])
df_metrics.to_csv('Data/performance_metrics.csv', index=False)

# 保存预测概率和标签
df_predictions = pd.DataFrame({
    'Probabilities': Probabilities,
    'Labels': y_test
})
df_predictions.to_csv('Data/probabilities.csv', index=False)
