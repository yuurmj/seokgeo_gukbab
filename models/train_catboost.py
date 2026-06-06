import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "decision"
ID_COLUMN = "pole_id"
CATEGORICAL_COLUMNS = ["region_type", "land_cover"]


def train_catboost(csv_path):
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

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN]

    for column in numerical_columns:
        X[column] = X[column].fillna(X[column].median())

    for column in categorical_columns:
        X[column] = X[column].fillna("missing").astype(str)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
    )
    prediction = model.predict(X_valid).astype(int).reshape(-1)

    print("Accuracy:", accuracy_score(y_valid, prediction))
    print("Precision:", precision_score(y_valid, prediction, zero_division=0))
    print("Recall:", recall_score(y_valid, prediction, zero_division=0))
    print("F1 Score:", f1_score(y_valid, prediction, zero_division=0))

    return model


if __name__ == "__main__":
    train_catboost("data/final_dataset.csv")
