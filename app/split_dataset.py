# import os
# import random
# import shutil

# # Define paths
# all_data_dir = "Tensorflow/workspace/data/all_data"
# train_dir = "Tensorflow/workspace/data/train"
# validation_dir = "Tensorflow/workspace/data/validation"
# test_dir = "Tensorflow/workspace/data/test"

# # Split ratios
# train_ratio = 0.7
# validation_ratio = 0.0
# test_ratio = 0.3

# # Gather all YAML and corresponding image paths
# yaml_files = [f for f in os.listdir(all_data_dir) if f.endswith('.yaml')]
# image_files = [f for f in os.listdir(all_data_dir) if f.endswith(('.jpg', '.png'))]

# # Map YAML files to corresponding images
# data_pairs = []
# for yaml_file in yaml_files:
#     image_file = os.path.splitext(yaml_file)[0] + '.jpg'
#     if image_file in image_files:
#         data_pairs.append((yaml_file, image_file))

# # Shuffle and split the data
# random.shuffle(data_pairs)
# total = len(data_pairs)

# train_count = int(total * train_ratio)
# validation_count = int(total * validation_ratio)

# train_data = data_pairs[:train_count]
# validation_data = data_pairs[train_count:train_count + validation_count]
# test_data = data_pairs[train_count + validation_count:]

# # Function to copy data to respective folders
# def copy_data(data, target_dir):
#     os.makedirs(target_dir, exist_ok=True)
#     for yaml_file, image_file in data:
#         shutil.copy(os.path.join(all_data_dir, yaml_file), target_dir)
#         shutil.copy(os.path.join(all_data_dir, image_file), target_dir)

# # Copy data to train, validation, and test folders
# copy_data(train_data, train_dir)
# copy_data(validation_data, validation_dir)
# copy_data(test_data, test_dir)

# print("Data split completed!")
# print(f"Train: {len(train_data)}")
# print(f"Validation: {len(validation_data)}")
# print(f"Test: {len(test_data)}")


import os
import random
import shutil

# Define paths
all_data_dir = "sar-ship-detection/yolov5/dataset/ship_dataset_v0"
train_dir = "sar-ship-detection/yolov5/dataset/train"
validation_dir = "sar-ship-detection/yolov5/dataset/validation"
test_dir = "sar-ship-detection/yolov5/dataset/test"

# Split ratios
train_ratio = 0.7
validation_ratio = 0.0
test_ratio = 0.3

# Gather all .txt and corresponding image paths
txt_files = [f for f in os.listdir(all_data_dir) if f.endswith('.txt')]
image_files = [f for f in os.listdir(all_data_dir) if f.endswith(('.jpg', '.png'))]

# Map .txt files to corresponding images
data_pairs = []
for txt_file in txt_files:
    image_file = os.path.splitext(txt_file)[0] + '.jpg'
    if image_file in image_files:
        data_pairs.append((txt_file, image_file))

# Shuffle and split the data
random.shuffle(data_pairs)
total = len(data_pairs)

train_count = int(total * train_ratio)
validation_count = int(total * validation_ratio)

train_data = data_pairs[:train_count]
validation_data = data_pairs[train_count:train_count + validation_count]
test_data = data_pairs[train_count + validation_count:]

# Function to copy data to respective folders
def copy_data(data, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for txt_file, image_file in data:
        shutil.copy(os.path.join(all_data_dir, txt_file), target_dir)
        shutil.copy(os.path.join(all_data_dir, image_file), target_dir)

# Copy data to train, validation, and test folders
copy_data(train_data, train_dir)
copy_data(validation_data, validation_dir)
copy_data(test_data, test_dir)

print("Data split completed!")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(validation_data)}")
print(f"Test: {len(test_data)}")
