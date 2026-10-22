from pathlib import Path
import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "loan_default_validation_model.joblib"
)

model = joblib.load(MODEL_PATH)

print("\n--- MODEL INFORMATION ---")
print("Model type:", type(model).__name__)
print("Model class:", type(model))

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f"{name}: {type(step).__name__}")

if hasattr(model, "feature_names_in_"):
    print("\nInput feature names:")
    print(list(model.feature_names_in_))

if hasattr(model, "get_params"):
    print("\nModel parameters:")
    print(model.get_params(deep=False))