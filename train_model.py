from src.churn_backend import train_and_save_model


if __name__ == "__main__":
    bundle = train_and_save_model(force=True)
    print("Saved model artifact.")
    print(f"Model: {bundle.metadata.get('model_name')}")
    print(f"Threshold: {bundle.threshold}")
    print(f"Metrics: {bundle.metadata.get('metrics')}")
