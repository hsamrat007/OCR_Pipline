import cv2
import os

def crop_left_bottom(image_path, left_crop_percent=10, bottom_crop_percent=10):
  
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not open image: {image_path}")

    h, w, _ = image.shape  

 
    crop_x = int(w * left_crop_percent / 100)  
    crop_y = int(h * bottom_crop_percent / 100)  

    
    cropped_image = image[:h - crop_y, crop_x:w]

    return cropped_image

def process_images_in_directory(directory, left_crop_percent=10, bottom_crop_percent=10):
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process images
            image_path = os.path.join(directory, filename)

            try:
                cropped_image = crop_left_bottom(image_path, left_crop_percent, bottom_crop_percent)

                # Overwrite the original image
                cv2.imwrite(image_path, cropped_image)
                print(f"Processed and replaced: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Set your directory and crop percentages based on the page remove extra borders or unnecessary information.
image_directory = "./"
left_crop_percent = 7  
bottom_crop_percent = 7  


process_images_in_directory(image_directory, left_crop_percent, bottom_crop_percent)
