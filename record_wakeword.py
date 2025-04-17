import openwakeword
import os

# Path to save the model
model_save_path = "C:\\AI\\Assistant\\models\\rex.onnx"  # Adjust path as needed

# Initialize the OpenWakeWord model with a custom backend
model = openwakeword.Model(backend="onnx")

# Record the wakeword "rex"
model.record_wakeword("rex", model_save_path)

print(f"Wakeword model saved to {model_save_path}")

