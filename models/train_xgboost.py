import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


TARGET_COLUMN = "decision"
ID_COLUMN = "pole_id"
CATEGORICAL_COLUMNS = ["region_type", "land_cover"]


def train_xgboost(csv_path):
    data = pd.read_csv(csv_path)

    if TARGET_COLUMN not in data.columns:
        print(f"{TARGET_COLUMN} 컬럼이 없어 모델 학습을 건너뜁니다.")
        return None

    feature_columns = [
        column
        for column in data.columns
        if column not in [TARGET_COLUMN, ID_COLUMN]
    ]
    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in feature_columns
    ]
    numerical_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    X = data[feature_columns]
    y = data[TARGET_COLUMN]

    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)

    print("Accuracy:", accuracy_score(y_valid, prediction))
    print("Precision:", precision_score(y_valid, prediction, zero_division=0))
    print("Recall:", recall_score(y_valid, prediction, zero_division=0))
    print("F1 Score:", f1_score(y_valid, prediction, zero_division=0))

    return model


if __name__ == "__main__":
    train_xgboost("data/final_dataset.csv")
