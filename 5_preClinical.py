import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

# 1. 加载数据 (保持不变)
try:
    df = pd.read_csv("clinical.csv")
    print(f"Loaded 'clinical.csv'. Total samples: {len(df)}")
except Exception as e:
    print(f"Error loading 'clinical.csv': {e}")
    sys.exit()

# 清理列名中的空格
df.columns = df.columns.str.strip()
print(f"Stripped columns: {list(df.columns)}")

# 修复 M 列中的错别字 (保持不变)
if 'M' in df.columns:
    print(f"Original unique 'M' values: {df['M'].unique()}")
    df['M'] = df['M'].astype(str).str.replace('MO', 'M0', case=False)
    print(f"Fixed 'M' column. New unique values: {df['M'].unique()}")

# 2. 定义特征列 (保持不变)
try:
    continuous_features = ['Age', 'ALB', 'HB', 'NLR', 'PLR', 'MONO']
    categorical_features = ['Sex', 'T', 'N', 'M']
    survival_labels = ['State', 'Time']
    id_col = 'Num'
    stratify_col = 'State'

    all_needed_cols = continuous_features + categorical_features + survival_labels + [id_col]
    missing_cols = [col for col in all_needed_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing: {missing_cols}")
        sys.exit()
    print("Column names successfully validated.")
except Exception as e:
    print(f"Error defining feature lists: {e}")
    sys.exit()

# 3. 严格划分数据集 (修改点：直接划分为 训练集150 / 测试集37)
print("Splitting data into Train (for 5-fold CV) and Test sets...")
try:
    # 这里的 test_size=37 确保测试集精确为37例，剩余的(150例)归入训练集
    df_train, df_test = train_test_split(
        df,
        test_size=37,
        stratify=df[stratify_col],
        random_state=42
    )

    print(f"Training set (for 5-fold CV): {len(df_train)} samples")  # 期望 150
    print(f"Test set (Hold-out): {len(df_test)} samples")  # 期望 37
    print("-" * 30)

except Exception as e:
    print(f"Error during data splitting: {e}")
    sys.exit()

# 4. 定义预处理流程 (保持不变)
continuous_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5. 使用 ColumnTransformer 组合 (保持不变)
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_pipeline, continuous_features),
        ('categorical', categorical_pipeline, categorical_features)
    ],
    remainder='drop'
)

# 6. 拟合预处理器 (保持不变，仅在训练集上Fit)
print("Fitting preprocessing pipeline ON TRAINING DATA ONLY...")
preprocessor.fit(df_train)
print("Fit complete.")

# 7. 应用预处理器 (修改点：移除了 val 的 transform)
train_features_processed = preprocessor.transform(df_train)
test_features_processed = preprocessor.transform(df_test)

# 8. 获取处理后的特征名称 (保持不变)
try:
    ohe_feature_names = preprocessor.named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(
        categorical_features)
except AttributeError:
    ohe_feature_names = preprocessor.named_transformers_['categorical']['encoder'].get_feature_names_out(
        categorical_features)

all_feature_names = continuous_features + list(ohe_feature_names)
print(f"Processed feature vector size: {len(all_feature_names)}")


# 9. 重新构建 DataFrame 并保存 (修改点：移除了 val 的构建)
def create_processed_df(original_df, processed_features, feature_names, id_col, label_cols):
    df_original_reset = original_df.reset_index(drop=True)
    df_features = pd.DataFrame(processed_features, columns=feature_names)
    df_labels_ids = df_original_reset[label_cols + [id_col]]
    df_final = pd.concat([df_labels_ids, df_features], axis=1)
    return df_final


df_train_processed = create_processed_df(df_train, train_features_processed, all_feature_names, id_col, survival_labels)
df_test_processed = create_processed_df(df_test, test_features_processed, all_feature_names, id_col, survival_labels)

# 10. 保存到 CSV (修改点：只保存 train 和 test)
# 这个 train_processed.csv (150例) 将用于你的 5-fold CV 脚本
df_train_processed.to_csv("train.csv", index=False)
# 这个 test_processed.csv (37例) 是最终独立的测试集
df_test_processed.to_csv("test.csv", index=False)

print("-" * 30)
print("Preprocessing and splitting complete.")
print("Saved files: 'train.csv' (150 samples), 'test.csv' (37 samples)")
print("Data has been cleaned (e.g., 'M' column typos fixed).")
