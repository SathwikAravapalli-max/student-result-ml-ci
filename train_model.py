
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    rng = np.random.default_rng(42)
    number_of_students = 300

    data = pd.DataFrame({
        "attendance": rng.integers(50, 101, number_of_students),
        "internal_marks": rng.integers(20, 101, number_of_students),
        "assignment_marks": rng.integers(30, 101, number_of_students),
        "previous_score": rng.integers(30, 101, number_of_students)
    })

    data["weighted_score"] = (
        0.25 * data["attendance"]
        + 0.35 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.20 * data["previous_score"]
    )

    # 1 = PASS, 0 = FAIL
    data["result"] = (data["weighted_score"] >= 60).astype(int)
    return data


def train_model():
    print("Creating dataset...")
    data = create_dataset()
    data.to_csv("student_results.csv", index=False)

    print("Dataset created successfully.")
    print("Number of records:", len(data))

    features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    X = data[features]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "student_result_model.pkl")
    print("\nModel saved as student_result_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
