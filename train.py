from ultralytics import YOLO
from pathlib import Path
import json
import glob
def convert_yolo_format(x1, y1, x2, y2, img_width, img_height):
    # Calculate center coordinates, width, and height in pixels
    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + (box_width / 2)
    y_center = y1 + (box_height / 2)
    
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    box_width /= img_width
    box_height /= img_height
    
    return x_center, y_center, box_width, box_height

# Load the YOLO model
model = YOLO("best.pt")

# Directory containing the images
image_dir = Path("tests")  # Update this path
output_data = []

# Iterate over all images in the directory
for img_path in glob.glob("*.jpg"):  # Change to '*.png' if needed
    print(img_path)
    results = model.predict(img_path)
    print(f"Results for {img_path}: {results}")  # Debugging

    # Get image dimensions
    img_width = 1024  # Replace with actual width if known
    img_height = 768  # Replace with actual height if known
    filename = img_path

    objects = []

    # Process each result
    for result in results:
        if result.boxes is None:
            print("No boxes detected.")  # Debugging
            continue

        for box in result.boxes:
            # Extract coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right corners
            confidence = box.conf[0]  # Confidence score
            class_id = box.cls[0]  # Class ID

            # Convert to normalized YOLO format
            x_center, y_center, box_width, box_height = convert_yolo_format(x1, y1, x2, y2, img_width, img_height)

            # Append the object details
            objects.append({
                "height": f"{box_height:.6f}",
                "obj_class": str(int(class_id)),  # Convert class ID to string
                "width": f"{box_width:.6f}",
                "x": f"{x_center:.6f}",
                "y": f"{y_center:.6f}"
            })

    if objects:  # Only add image data if objects were detected
        # Construct the data for this image
        image_data = {
            "filename": filename,
            "objects": objects
        }
        output_data.append(image_data)

# Convert the output data to JSON format
if output_data:  # Check if there is any output data to save
    json_data = json.dumps(output_data, indent=4)

    # Save the JSON data to a file
    with open("predictions.json", "w") as json_file:
        json_file.write(json_data)
    print("Predictions saved to predictions.json")
else:
    print("No predictions to save.")