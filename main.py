# main.py

from model import load_data, preprocess_data, build_model, train_model, evaluate_model
import joblib


def main():
    print("----- ANN Bank Churn Prediction -----")

    data_path = "Artificial_Neural_Network_Case_Study_data.csv"

    # Step 1: Load Data
    data = load_data(data_path)

    # ✅ CHANGE HERE (IMPORTANT)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    # Step 3: Build Model
    model = build_model(X_train.shape[1])
    model.summary()

    # Step 4: Train Model
    train_model(model, X_train, y_train)

    # Step 5: Evaluate Model
    evaluate_model(model, X_test, y_test)

    # ✅ ADD THIS PART (SAVE FILES)
    model.save("churn_model.keras")
    joblib.dump(sc, "scaler.pkl")
    joblib.dump(ct, "encoder.pkl")

    print("✅ Files saved successfully")


if __name__ == "__main__":
    main()