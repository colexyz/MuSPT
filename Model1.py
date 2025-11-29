import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------
# 设置中文字体（可选）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

#-------------------------------------------------
# 1. 数据处理 —— 数据集 A（原始数据，用于训练模型）
#-------------------------------------------------
file_path_A = "Data/Study1_data.csv" #424个被试
df_A = pd.read_csv(file_path_A)



# 人格问卷数据
cbf_df_A = df_A.copy()
# 音乐评分原始
rating_cols = [
    "您对音频A的感受评价是？", "您对音频B的感受评价是？", "您对音频C的感受评价是？",
    "您对音频D的感受评价是？", "您对音频E的感受评价是？",
    "您对音频F的感受评价是？", "您对音频G的感受评价是？", "您对音频H的感受评价是？",
    "您对音频I的感受评价是？", "您对音频J的感受评价是？",
    "您对音频K的感受评价是？", "您对音频L的感受评价是？", "您对音频M的感受评价是？",
    "您对音频N的感受评价是？", "您对音频O的感受评价是？",
    "您对音频P的感受评价是？", "您对音频Q的感受评价是？", "您对音频R的感受评价是？",
    "您对音频S的感受评价是？", "您对音频T的感受评价是？",
    "您对音频U的感受评价是？", "您对音频V的感受评价是？", "您对音频W的感受评价是？",
    "您对音频X的感受评价是？", "您对音频Y的感受评价是？"
]
ratings_raw_A = df_A[rating_cols].values.astype(np.float32)
ratings_converted_A = ratings_raw_A - 4.0  # 转至 [-3,3]

# ----------------------------------
# 人格问卷反向计分 & 维度计算
# ----------------------------------
cbf_pi_b_columns = { "神经质": [
        "我常感到害怕",
        "有时我觉得自己一无是处",
        "别人一句漫不经心的话，我常会联系在自己身上",
        "在面对压力时，我有种快要崩溃的感觉",
        "我常担忧一些无关紧要的事情",
        "我常常感到内心不踏实",
        "我常担心有什么不好的事情要发生",
        "我很少感到忧郁或沮丧"  # 反向计分
    ],
    "严谨性": [
        "一旦确定了目标，我会坚持努力地实现它",
        "我常常是仔细考虑之后才做出决定",
        "别人认为我是个慎重的人",
        "我喜欢一开头就把事情计划好",
        "我工作或学习很勤奋",
        "我是个倾尽全力做事的人",
        "在工作上，我常只求能应付过去便可",  # 反向计分
        "做事讲究逻辑和条理是我的一个特点"
    ],
    "宜人性": [
        "我觉得大部分人基本上是心怀善意的",
        "我不太关心别人是否受到不公正的待遇",         # 反向计分
        "我时常觉得别人的痛苦与我无关",            # 反向计分
        "我是那种只照顾好自己，不替别人担忧的人",      # 反向计分
        "虽然社会上有些骗子，但我觉得大部分人还是可信的",
        "当别人向我诉说不幸时，我常感到难过",
        "尽管人类社会存在着一些阴暗的东西（如战争、罪恶、欺诈），我仍然相信人性总的来说是善良的",
        "我常为那些遭遇不幸的人感到难过"
    ],
    "开放性": [
        "我头脑中经常充满生动的画面",
        "我是个勇于冒险，突破常规的人",
        "我喜欢冒险",
        "我对许多事情有着很强的好奇心",
        "我身上具有别人没有的冒险精神",
        "我渴望学习一些新东西，即使它们与我的日常生活无关",
        "我的想象力相当丰富",
        "我很愿意也很容易接受那些新事物、新观点、新想法"
    ],
    "外向性": [
        "我对人多的聚会感到乏味",           # 反向计分
        "在热闹的聚会上，我常常表现主动并尽情玩耍",
        "我尽量避免参加人多的聚会和嘈杂的环境",  # 反向计分
        "在一个团体中，我希望处于领导地位",
        "我希望成为领导者而不是被领导者",
        "别人多认为我是一个热情和友好的人",
        "我喜欢参加社交与娱乐聚会",
        "我希望成为领导者而不是被领导者"
    ]}
reverse_items_personality = ["我对人多的聚会感到乏味",
    "我不太关心别人是否受到不公正的待遇",
    "我时常觉得别人的痛苦与我无关",
    "我尽量避免参加人多的聚会和嘈杂的环境",
    "我是那种只照顾好自己，不替别人担忧的人",
    "在工作上，我常只求能应付过去便可",
    "我很少感到忧郁或沮丧"]
# 对反向题目进行 7-原分
for col in reverse_items_personality:
    if col in cbf_df_A.columns:
        cbf_df_A[col] = 7 - cbf_df_A[col]
# 各维度均值 & 归一化
dims = ["神经质", "严谨性", "宜人性", "开放性", "外向性"]
for dim, items in cbf_pi_b_columns.items():
    cbf_df_A[dim] = cbf_df_A[items].mean(axis=1)
for dim in dims:
    cbf_df_A[dim] = (cbf_df_A[dim] - 1) / 5.0

#--------------------------------------------
# 2. 数据处理 —— 数据集 B（外部测试）
#--------------------------------------------
file_path_B = "Data/Study2_data.csv"
df_B = pd.read_csv(file_path_B)
cbf_df_B = df_B.copy()
ratings_raw_B = df_B[rating_cols].values.astype(np.float32)
ratings_converted_B = ratings_raw_B - 4.0




# 反向计分
for col in reverse_items_personality:
    if col in cbf_df_B.columns:
        cbf_df_B[col] = 7 - cbf_df_B[col]
# 维度计算 & 归一化
for dim, items in cbf_pi_b_columns.items():
    cbf_df_B[dim] = cbf_df_B[items].mean(axis=1)
for dim in dims:
    cbf_df_B[dim] = (cbf_df_B[dim] - 1) / 5.0





#------------------------------------------------------------------------
# 自动循环训练：mean, mean_std，full
#------------------------------------------------------------------------
#定义三种方式 - 统计特征函数：mean, mean_std，full
def group_statistics(ratings, stat_type='mean'):
    """
    ratings: ndarray, shape (n_samples, 25)
    stat_type: 'sum', 'mean', or 'mean_std'
    """
    group_indices = [list(range(i, i+5)) for i in range(0, 25, 5)]
    features = []

    for idx in group_indices:
        group = ratings[:, idx]  
        if stat_type == 'mean':
            feat = np.mean(group, axis=1, keepdims=True)
        elif stat_type == 'mean_std':
            mean_ = np.mean(group, axis=1, keepdims=True)
            std_  = np.std(group, axis=1, ddof=0, keepdims=True)
            feat = np.concatenate([mean_, std_], axis=1)
        elif stat_type == 'full':
            # 改动部分：统计1-7的频率分布
            # 输出 shape: (n_samples, 7)
           feat = np.array([(group == b).sum(axis=1) / group.shape[1] for b in range(-3, 4)]).T
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")
        features.append(feat)
    return np.concatenate(features, axis=1)

feature_versions = {
     'mean':     {'type': 'mean',     'input_dim': 5},
     'mean_std': {'type': 'mean_std', 'input_dim': 10},
    'full':     {'type': 'full',     'input_dim': 35}
}



#---------------------------------------------------------------
# 模型定义（单隐藏层）
#---------------------------------------------------------------
class SimpleNNRegressor(nn.Module):
    
    def __init__(self, input_dim=5, hidden_dim1=16, hidden_dim2=16, hidden_dim3=16,
                 hidden_dim4=16, hidden_dim5=8, hidden_dim6=8,hidden_dim7=8, 
                 output_dim=1, dropout_prob=0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob) 
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.bn4 = nn.BatchNorm1d(hidden_dim4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=dropout_prob)
        
        self.fc5 = nn.Linear(hidden_dim4, hidden_dim5)
        self.bn5 = nn.BatchNorm1d(hidden_dim5)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=dropout_prob)
        
        self.fc6 = nn.Linear(hidden_dim5, hidden_dim6)
        self.bn6 = nn.BatchNorm1d(hidden_dim6)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=dropout_prob)
        
        self.fc7 = nn.Linear(hidden_dim6, hidden_dim7)
        self.bn7 = nn.BatchNorm1d(hidden_dim7)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=dropout_prob)
        

        self.fc8 = nn.Linear(hidden_dim7, output_dim)
        


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.dropout7(x)

        logit = self.fc8(x)
        return logit

# -------------------------------------------------
# 三种基线模型
# -------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np

def baseline_random(y_true):
    #随机预测：在[0,1]之间均匀采样
    np.random.seed(42)
    return np.random.rand(len(y_true))

def baseline_constant(y_true, const=0.5):
    #定值预测：恒等于0.5
    return np.full(len(y_true), const)

def baseline_group_mean_cv(y, kf):
    #群体均值预测（十折交叉）
    preds = np.zeros_like(y, dtype=np.float32)
    for tr_idx, val_idx in kf.split(y):
        mean_val = np.mean(y[tr_idx])
        preds[val_idx] = mean_val
    return preds


#-------------------------------------------------------------------------
#训练的准备
#-------------------------------------------------------------------------
#控制随机性
def set_seed(seed=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 即使用不到 GPU，保险起见保留
    torch.backends.cudnn.benchmark = False

set_seed(25)


#--------------------------------------------------------------------------
#主循环开始
#---------------------------------------------------------------------------
save_pred_dir = 'Detailed_Output/'
final_results = {}
final_results1 = {}
prediction_logs = []
A_preds_dict = {}

B_preds_dict = {}
errors = {name: {} for name in feature_versions.keys()}
baseline_names = ['baseline_random', 'baseline_constant', 'baseline_mean_cv']
for b in baseline_names:
    errors[b] = {}

errors_B = {name: {} for name in feature_versions.keys()}
for b in baseline_names:
    errors_B[b] = {}


for v_name, cfg in feature_versions.items():##输入方式的循环###################################
    ##数据的准备
    X_A_feat = group_statistics(ratings_converted_A, stat_type=cfg['type']).astype(np.float32)
    X_B_feat = group_statistics(ratings_converted_B, stat_type=cfg['type']).astype(np.float32)
    results = {}
    results1 = {}


    ##人格维度的循环##########################################################################
    for dim in dims:
        y_A = cbf_df_A[dim].values.astype(np.float32)
        y_B = cbf_df_B[dim].values.astype(np.float32)

        ## 引入10折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        
        ##折的循环############################################################################
        val_preds_all_folds =[0] * 424
        train_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_A_feat)):
            X_tr, X_val = X_A_feat[train_idx], X_A_feat[val_idx]
            y_tr, y_val = y_A[train_idx], y_A[val_idx]
            tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
            val_ds= TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
            tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True)

            #模型初始化
            model = SimpleNNRegressor(input_dim=cfg['input_dim'])#模型
            opt = optim.Adam(model.parameters(), lr=0.01)#优化器和学习率
            crit= nn.BCEWithLogitsLoss()#损失函数

            
            epoch_losses = []
            
            #模型训练循环，每一折跑 1000 个 epoch#####################################################################################
            for ep in range(1000): 
                model.train()
                total_loss = 0
                for xb, yb in tr_ld:
                    opt.zero_grad()
                    loss = crit(model(xb), yb.unsqueeze(1))
                    loss.backward()
                    opt.step()

        
            # 该折的验证##############################################################################################
            model.eval()#切换到“评估模式”：Dropout 和 BatchNorm 行为会改变（Dropout 不再随机遮盖）更稳定、真实反映模型性能
            with torch.no_grad():
                pred_val = torch.sigmoid(model(torch.tensor(X_val))).squeeze().numpy()
                pred_train = torch.sigmoid(model(torch.tensor(X_tr))).squeeze().numpy()
                for idx, i in enumerate(val_idx):
                    val_preds_all_folds[i] = pred_val[idx]
            
            rmse_train = np.sqrt(mean_squared_error(y_tr, pred_train))
            corr_train = np.corrcoef(y_tr, pred_train)[0, 1]
            train_metrics.append([rmse_train, corr_train])
        
        # 评估
        avg_train_rmse, avg_train_corr = np.mean(train_metrics, axis=0)
        rmse_v = np.sqrt(mean_squared_error(y_A, val_preds_all_folds))
        # r2_v   = r2_score(y_A, val_preds_all_folds)
        corr_v = np.corrcoef(y_A, val_preds_all_folds)[0, 1]
        abs_errors = np.abs(y_A - val_preds_all_folds)
        errors[v_name][dim] = abs_errors.astype(np.float32)


        # 训练全集模型用于B测试（保证和原始一致，10折不影响B评估）
        model = SimpleNNRegressor(input_dim=cfg['input_dim'])
        opt = optim.Adam(model.parameters(), lr=0.01)
        crit= nn.BCEWithLogitsLoss()
        
        full_tr_ds = TensorDataset(torch.tensor(X_A_feat), torch.tensor(y_A))
        full_tr_ld = DataLoader(full_tr_ds, batch_size=256, shuffle=True)
        
        for ep in range(1000):
            model.train()
            for xb, yb in full_tr_ld:
                opt.zero_grad()
                loss = crit(model(xb), yb.unsqueeze(1))
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred_B = torch.sigmoid(model(torch.tensor(X_B_feat))).squeeze().numpy()
        

        mse_B = mean_squared_error(y_B, pred_B)
        rmse_B= np.sqrt(mse_B)
        corr_B= np.corrcoef(y_B, pred_B)[0,1]
   
        # # ---------- 三个基线 ----------
        # # 随机预测
        rand_preds = baseline_random(y_A)
        rmse_rand = np.sqrt(mean_squared_error(y_A, rand_preds))
        corr_rand = np.corrcoef(y_A, rand_preds)[0,1]

        # 定值预测 0.5
        const_preds = baseline_constant(y_A, const=0.5)
        rmse_const = np.sqrt(mean_squared_error(y_A, const_preds))
        corr_const = np.corrcoef(y_A, const_preds)[0,1]

        # 群体均值预测（十折CV）
        mean_cv_preds = baseline_group_mean_cv(y_A, kf)
        rmse_mean_cv = np.sqrt(mean_squared_error(y_A, mean_cv_preds))
        corr_mean_cv = np.corrcoef(y_A, mean_cv_preds)[0,1]

      
        errors['baseline_random'][dim] = np.abs(y_A - rand_preds).astype(np.float32)
        errors['baseline_constant'][dim] = np.abs(y_A - const_preds).astype(np.float32)
        errors['baseline_mean_cv'][dim] = np.abs(y_A - mean_cv_preds).astype(np.float32)
      
      
      
        # ---------- B 集三种基线 ----------
        rand_preds_B = baseline_random(y_B)
        rmse_rand_B = np.sqrt(mean_squared_error(y_B, rand_preds_B))
        corr_rand_B = np.corrcoef(y_B, rand_preds_B)[0,1]

        const_preds_B = baseline_constant(y_B, const=0.5)
        rmse_const_B = np.sqrt(mean_squared_error(y_B, const_preds_B))
        corr_const_B = np.corrcoef(y_B, const_preds_B)[0,1]

        mean_B = np.mean(y_A)  # 注意用 A 的均值来预测 B
        pred_B_mean = np.full(len(y_B), mean_B)
        rmse_B_mean = np.sqrt(mean_squared_error(y_B, pred_B_mean))
        corr_B_mean = np.corrcoef(y_B, pred_B_mean)[0,1]

        # errors_B 使用与 errors 相同的 key 体系（feature_versions keys + baseline_names）
        errors_B[v_name][dim] = np.abs(y_B - pred_B).astype(np.float32)
        errors_B['baseline_random'][dim] = np.abs(y_B - rand_preds_B).astype(np.float32)
        errors_B['baseline_constant'][dim] = np.abs(y_B - const_preds_B).astype(np.float32)
        # 将 A 的均值预测误差也命名为 baseline_mean_cv，保持宽表列名对称（虽然计算来源不同）
        errors_B['baseline_mean_cv'][dim] = np.abs(y_B - pred_B_mean).astype(np.float32)


        #---------- 保存 ----------
        results1[dim] = {
            # A 集
            'A随机_RMSE': rmse_rand, 'A随机_Corr': corr_rand,
            'A常_RMSE': rmse_const, 'A常_Corr': corr_const,
            'A平均_RMSE': rmse_mean_cv, 'A平均_Corr': corr_mean_cv,
            # B 集
            'B随机_RMSE': rmse_rand_B, 'B随机_Corr': corr_rand_B,
            'B常_RMSE': rmse_const_B, 'B常_Corr': corr_const_B,
            'B平均_RMSE': rmse_B_mean, 'B平均_Corr': corr_B_mean,
        }
        
        results[dim] = {
            'A_Train_RMSE': avg_train_rmse, 'A_Train_Corr': avg_train_corr,
            'A_Val_RMSE': rmse_v, 'A_Val_Corr': corr_v,
            'B_Test_RMSE': rmse_B, 'B_Test_Corr': corr_B
        }
        A_preds_dict[dim] = val_preds_all_folds
        B_preds_dict[dim] = pred_B

    df_A_preds = pd.DataFrame(A_preds_dict)  # 行 -> A 样本，列 -> 维度名
    df_B_preds = pd.DataFrame(B_preds_dict)  # 行 -> B 样本，列 -> 维度名


    final_results[v_name] = pd.DataFrame.from_dict(results, orient='index')
    final_results1[v_name] = pd.DataFrame.from_dict(results1, orient='index')


# %% 打印所有版本结果
output_path = "Main_Result/Model1_result.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for vn, df in final_results.items():
        f.write(f"\n=== 版本 {vn} 最终结果 ===\n")
        f.write(df.round(4).to_string())
        f.write("\n\n")



# %% 打印基线模型结果
output_path1 = "Main_Result/Model1_result_Baseline.txt"


with open(output_path1, "w", encoding="utf-8") as f:
    for vn, df in final_results1.items():
        f.write(f"\n=== 版本 {vn} 基线模型结果 ===\n")
        f.write(df.round(4).to_string())
        f.write("\n\n")

# 将收集到的 errors 转为宽表（subject_id, trait, mean, mean_std, full）
# ---------------------------
rows = []
model_names = list(feature_versions.keys()) + baseline_names
n_samples =424
df_ids = [f"subj_{i+1}" for i in range(n_samples)]
for i in range(n_samples):
    for dim in dims:
        row = {
            'subject_id': df_ids[i],
            'trait': dim
        }
        for m in model_names:
            # 取值并转为 Python float（方便 CSV）
            row[m] = float(errors[m][dim][i])
        rows.append(row)

df_wide = pd.DataFrame(rows)

# 指定列顺序：subject_id, trait, 然后各模型
col_order = ['subject_id', 'trait'] + model_names
df_wide = df_wide[col_order]

# 保存 CSV（UTF-8-sig 方便 Excel 打开）
out_path = os.path.join(save_pred_dir, "A_wide_abs_errors_model1.csv")
df_wide.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"保存完成：{out_path}")




rows_B = []
model_names_B = list(feature_versions.keys()) + baseline_names
n_samples_B = cbf_df_B.shape[0]
df_ids_B = [f"subj_{i+1}" for i in range(n_samples_B)]
for i in range(n_samples_B):
    for dim in dims:
        row = {
            'subject_id': df_ids_B[i],
            'trait': dim
        }
        for m in model_names_B:
            row[m] = float(errors_B[m][dim][i])
        rows_B.append(row)

df_wide_B = pd.DataFrame(rows_B)
col_order_B = ['subject_id', 'trait'] + model_names_B
df_wide_B = df_wide_B[col_order_B]

out_path_B = os.path.join(save_pred_dir, "B_wide_abs_errors_model1.csv")
df_wide_B.to_csv(out_path_B, index=False, encoding="utf-8-sig")
print(f"保存完成：{out_path_B}")