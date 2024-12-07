import yaml
import os

# Define class mapping
classes = {f"ship-{i}": 0 for i in range(1, 31)} # Map labels to class IDs (extend as needed)

# Directory containing YAML files
yaml_dir = "all_data"  # Replace with your folder path containing YAML files
output_dir = "all_data"  # Directory to save YOLO TXT files
os.makedirs(output_dir, exist_ok=True)

# Loop through all YAML files in the directory
for yaml_file in os.listdir(yaml_dir):
    if yaml_file.endswith(".yaml"):
        yaml_path = os.path.join(yaml_dir, yaml_file)

        # Load YAML file
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract image dimensions
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]

        # Prepare YOLO format data
        yolo_annotations = []
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            # Extract bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            # Normalize coordinates
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            box_width = (xmax - xmin) / img_width
            box_height = (ymax - ymin) / img_height

            # Get class ID
            class_id = classes[label]

            # Append annotation
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

        # Save to TXT file
        txt_filename = os.path.splitext(yaml_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_annotations))
        os.remove(yaml_path)
        print(f"Annotations saved to {txt_path}")



# import os
# import shutil

# # Source and destination folders
# source_folder = "dataset/images/train"  # Replace with the path to your source folder
# destination_folder = "dataset/labels/train"  # Replace with the path to your destination folder

# # Ensure the destination folder exists
# os.makedirs(destination_folder, exist_ok=True)

# # Loop through files in the source folder
# for file_name in os.listdir(source_folder):
#     # Check if the file is a .txt file
#     if file_name.endswith(".txt"):
#         source_path = os.path.join(source_folder, file_name)
#         destination_path = os.path.join(destination_folder, file_name)
        
#         # Move the file
#         shutil.move(source_path, destination_path)
#         print(f"Moved: {file_name}")

# print("All .txt files have been moved.")
