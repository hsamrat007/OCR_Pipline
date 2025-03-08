import os
import shutil
import random

original_dir = "/teamspace/studios/this_studio/images "  
output_dir = "./"  
train_ratio = 0.8  

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


images = [f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(images)  


train_size = int(len(images) * train_ratio)
train_images = images[:train_size]
test_images = images[train_size:]

for img in train_images:
    shutil.copy(os.path.join(original_dir, img), os.path.join(train_dir, img))

for img in test_images:
    shutil.copy(os.path.join(original_dir, img), os.path.join(test_dir, img))

print(f"Dataset split complete! Train: {len(train_images)}, Test: {len(test_images)}")

