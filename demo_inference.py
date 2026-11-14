from ultralytics import YOLO
import os

# 1. Load the best model weights we just trained from the E6 experiment
model_path = "runs/detect/results/e6_context_refine/seed_42/weights/last.pt"
print(f"Loading our trained model from: {model_path} ...")
model = YOLO(model_path)

# 2. Pick a test image that the model has never seen during training
test_image = "images/test/xylem_0212.jpg"
print(f"Running prediction on: {test_image} ...")

# 3. Run inference! 
# save=True tells YOLO to draw the boxes and save the image.
results = model.predict(source=test_image, save=True, project="results", name="demo_inference")

# 4. Print out what it found
result = results[0]
print("\n--- INFERENCE COMPLETE ---")
print(f"Found {len(result.boxes)} plant cells in the image!")

vessel_count = 0
fiber_count = 0
parenchyma_count = 0

for box in result.boxes:
    class_id = int(box.cls[0].item())
    if class_id == 0: vessel_count += 1
    elif class_id == 1: fiber_count += 1
    elif class_id == 2: parenchyma_count += 1

print(f"- Vessels detected: {vessel_count}")
print(f"- Fibers detected: {fiber_count}")
print(f"- Parenchyma detected: {parenchyma_count}")
print(f"\nThe output image with bounding boxes has been saved to: {result.save_dir}")
