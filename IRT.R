library(mirt)
library(readr)


# 用于创建交叉验证划分
create_folds <- function(data, k = 10, seed = 42) {
  set.seed(seed)
  n <- nrow(data)
  # 随机打乱样本顺序
  shuffled_idx <- sample(1:n)
  # 把样本分成k折
  folds <- split(shuffled_idx, cut(seq_along(shuffled_idx), k, labels = FALSE))
  return(folds)
}

# 1. 读取原始数据
data_path <- "D:/Code_Data/Data/ratings_A.csv"
all_data <- read.csv(data_path)
out_dir <- "D:/Code_Data/IRT/"
cat("Creating 10-fold split and processing...\n")

# 2. 生成10折划分
folds <- create_folds(all_data, k = 10, seed = 42)

# ========== 十折交叉验证 ==========
for (fold in 1:10) {
  cat(sprintf("\n=== 正在处理 Fold %d ===\n", fold))
  
  # 获取当前fold的训练集与验证集
  test_idx <- folds[[fold]]
  train_data <- all_data[-test_idx, ]
  valid_data <- all_data[test_idx, ]
  
  # 转换为有序整数矩阵
  train_ord <- as.data.frame(lapply(train_data, as.integer))
  valid_ord <- as.data.frame(lapply(valid_data, as.integer))
  
  # 拟合 IRT 模型
  mod <- mirt(
    data = train_ord,
    model = 5,  # 五个潜变量（与人格维度数对应）
    itemtype = "graded",
    method = "QMCEM",
    exploratory = TRUE
  )
  
  # 导出题目参数
  write_csv(as.data.frame(coef(mod, simplify = TRUE)$items),
            file.path(out_dir, sprintf("fold%d_itempars.csv", fold)))
  
  # 导出训练集 theta
  fs_train <- fscores(mod, method = "EAP", QMC = TRUE)
  write_csv(as.data.frame(fs_train),
            file.path(out_dir, sprintf("fold%d_train_theta.csv", fold)))
  
  # 清理测试集响应
  for (i in seq_along(valid_ord)) {
    valid_vals <- unique(train_ord[[i]])
    valid_ord[[i]][!(valid_ord[[i]] %in% valid_vals)] <- NA
  }
  
  n_items <- ncol(train_ord)
  n_train <- nrow(fs_train)
  n_valid <- nrow(fs_valid)
  
  # ================== 训练集预测评分 ==================
  train_pred <- matrix(NA, nrow=n_train, ncol=n_items)
  for (i in 1:n_train) {
    theta_i <- fs_train[i, ]
    for (j in 1:n_items) {
      item_j <- extract.item(mod, j)
      p <- probtrace(item_j, Theta = theta_i)
      train_pred[i,j] <- sum(0:(length(p)-1) * p)
    }
  }
  train_pred_df <- as.data.frame(train_pred)
  write_csv(train_pred_df, file.path(out_dir, "train_pred_probtrace.csv"))
  
  # ================== 测试集预测评分 ==================
  valid_pred <- matrix(NA, nrow=n_valid, ncol=n_items)
  for (i in 1:n_valid) {
    theta_i <- fs_valid[i, ]
    for (j in 1:n_items) {
      p <- probtrace(mod, Theta=theta_i, which.items=j)
      valid_pred[i,j] <- sum(0:(length(p)-1) * p)
    }
  }
  valid_pred_df <- as.data.frame(valid_pred)
  write_csv(valid_pred_df, file.path(out_dir, "test_pred_probtrace.csv"))
  
  # ================== 指标计算 ==================
  train_rmse <- sapply(seq_along(train_ord), function(i) {
    Metrics::rmse(as.vector(train_ord[[i]]), as.vector(train_pred_df[[i]]))
  })
  train_cor <- sapply(seq_along(train_ord), function(i) {
    cor(as.vector(train_ord[[i]]), as.vector(train_pred_df[[i]]), use="pairwise.complete.obs")
  })
  
  valid_rmse <- sapply(seq_along(valid_ord), function(i) {
    Metrics::rmse(as.vector(valid_ord[[i]]), as.vector(valid_pred_df[[i]]))
  })
  valid_cor <- sapply(seq_along(valid_ord), function(i) {
    cor(as.vector(valid_ord[[i]]), as.vector(valid_pred_df[[i]]), use="pairwise.complete.obs")
  })
  
  cat(sprintf("Train RMSE: %s, Cor: %s\n", 
              paste(round(train_rmse,3), collapse=", "), 
              paste(round(train_cor,3), collapse=", ")))
  cat(sprintf("Valid RMSE: %s, Cor: %s\n", 
              paste(round(valid_rmse,3), collapse=", "), 
              paste(round(valid_cor,3), collapse=", ")))
}

# ========== 汇总输出 ==========
final_results <- do.call(rbind, all_fold_results)

# 输出 CSV
write_excel_csv(final_results, file.path(out_dir, "评分预测_10折指标.csv"))






##############################################################################

ibrary(tidyverse)
library(Metrics)

# ========== 路径设置 ==========
theta_dir <- "D:/Code_Data/Data/IRT/"
personality_path <- "D:/Code_Data/Data/cbf_df_A.csv"
rating_path <- "D:/Code_Data/Data/ratings_A.csv"

# ========== 读取数据 ==========
personality <- read.csv(personality_path)
ratings <- read.csv(rating_path)


# ========== 存储结果 ==========
all_personality_results <- list()
all_rating_results <- list()

# ========== 十折循环 ==========
for (fold in 1:10) {
  cat(sprintf("\n>>> 正在处理 Fold %d <<<\n", fold))
  
  # 读取该折的 theta 特征
  train_theta <- read.csv(file.path(theta_dir, sprintf("fold%d_train_theta.csv", fold)))
  test_theta  <- read.csv(file.path(theta_dir, sprintf("fold%d_test_theta.csv", fold)))
  
  # 获取索引
  test_idx <- folds[[fold]]
  train_idx <- setdiff(seq_len(nrow(personality)), test_idx)
  
  # 分割人格与评分数据
  train_personality <- personality[train_idx, ]
  test_personality  <- personality[test_idx, ]
  train_ratings <- ratings[train_idx, ]
  test_ratings  <- ratings[test_idx, ]
  
  # -------------------------------
  # （1）人格预测部分
  # -------------------------------
  fold_personality_results <- data.frame()
  
  for (dim_name in colnames(personality)) {
    model <- lm(train_personality[[dim_name]] ~ ., data = train_theta)
    preds <- predict(model, newdata = test_theta)
    true_vals <- test_personality[[dim_name]]
    
    rmse_val <- rmse(true_vals, preds)
    cor_val <- cor(true_vals, preds)
    
    fold_personality_results <- rbind(fold_personality_results, data.frame(
      Fold = fold,
      维度 = dim_name,
      RMSE = rmse_val,
      相关 = cor_val
    ))
  }
  all_personality_results[[fold]] <- fold_personality_results
  
  
  # -------------------------------
  # （2）评分预测部分
  # -------------------------------
  fold_rating_results <- data.frame()
  
  for (rating_name in colnames(ratings)) {
    model <- lm(train_ratings[[rating_name]] ~ ., data = train_theta)
    preds <- predict(model, newdata = test_theta)
    true_vals <- test_ratings[[rating_name]]
    
    rmse_val <- rmse(true_vals, preds)
    cor_val <- cor(true_vals, preds)
    
    fold_rating_results <- rbind(fold_rating_results, data.frame(
      Fold = fold,
      项目 = rating_name,
      RMSE = rmse_val,
      相关 = cor_val
    ))
  }
  all_rating_results[[fold]] <- fold_rating_results
}

# ========== 汇总输出 ==========
final_personality_results <- do.call(rbind, all_personality_results)
final_rating_results <- do.call(rbind, all_rating_results)

# RMSE宽表
wide_rmse <- final_personality_results |>
  select(Fold, 维度, RMSE) |>
  pivot_wider(names_from = 维度, values_from = RMSE, names_glue = "{维度}_RMSE")

# 相关宽表
wide_cor <- final_personality_results |>
  select(Fold, 维度, 相关) |>
  pivot_wider(names_from = 维度, values_from = 相关, names_glue = "{维度}_相关")

# 合并两者
wide_personality <- left_join(wide_rmse, wide_cor, by = "Fold")

# === 导出宽表 ===
write_excel_csv(wide_personality,
          file.path(theta_dir, "人格预测_10fold指标_宽表.csv"))

# 输出 CSV（UTF-8 编码，避免乱码）
write_excel_csv(final_personality_results,
          file.path(theta_dir, "人格预测_10fold指标.csv"))

write_excel_csv(final_rating_results,
          file.path(theta_dir, "评分预测_10fold指标.csv"))

View(final)
























