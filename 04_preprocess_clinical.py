"""
Preprocess clinical data and apply standardized train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

def main():
    clinical_csv = "./data/clinical.csv"
    try:
        df = pd.read_csv(clinical_csv)
    except Exception as e:
        print(f"Error loading {clinical_csv}: {e}")
        sys.exit()

    df.columns = df.columns.str.strip()

    if 'M' in df.columns:
        df['M'] = df['M'].astype(str).str.replace('MO', 'M0', case=False)

    continuous_features = ['Age', 'ALB', 'HB', 'NLR', 'PLR', 'MONO']
    categorical_features = ['Sex', 'T', 'N', 'M']
    survival_labels = ['State', 'Time']
    id_col = 'Num'
    stratify_col = 'State'

    df_train, df_test = train_test_split(
        df, test_size=37, stratify=df[stratify_col], random_state=42
    )

    continuous_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipeline, continuous_features),
            ('categorical', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    preprocessor.fit(df_train)

    train_features_processed = preprocessor.transform(df_train)
    test_features_processed = preprocessor.transform(df_test)

    try:
        ohe_feature_names = preprocessor.named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(categorical_features)
    except AttributeError:
        ohe_feature_names = preprocessor.named_transformers_['categorical']['encoder'].get_feature_names_out(categorical_features)

    all_feature_names = continuous_features + list(ohe_feature_names)

    def create_processed_df(original_df, processed_features, feature_names, id_col, label_cols):
        df_original_reset = original_df.reset_index(drop=True)
        df_features = pd.DataFrame(processed_features, columns=feature_names)
        df_labels_ids = df_original_reset[label_cols + [id_col]]
        return pd.concat([df_labels_ids, df_features], axis=1)

    df_train_processed = create_processed_df(df_train, train_features_processed, all_feature_names, id_col, survival_labels)
    df_test_processed = create_processed_df(df_test, test_features_processed, all_feature_names, id_col, survival_labels)

    df_train_processed.to_csv("./data/train_processed.csv", index=False)
    df_test_processed.to_csv("./data/test_processed.csv", index=False)

    print("Preprocessing and splitting complete.")
    print(f"Saved: train_processed.csv ({len(df_train)} samples), test_processed.csv ({len(df_test)} samples)")

if __name__ == "__main__":
    main()
