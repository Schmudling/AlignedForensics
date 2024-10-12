import os
import shutil

def process_images(image_dir, text_file, keep_dir, move_dir):
    # Ensure the output directories exist
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(move_dir, exist_ok=True)
    
    # Read the list of image names to keep
    with open(text_file, 'r') as file:
        keep_images = set(line.strip() for line in file)
    
    # Process each image in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            if image_name in keep_images:
                shutil.move(image_path, os.path.join(keep_dir, image_name))
            else:
                shutil.move(image_path, os.path.join(move_dir, image_name))

# Example usage
image_dir = '/nobackup3/anirudh/datasets/coco/val2017'
text_file = '/nobackup3/anirudh/DMimageDetection/training_code/latent_diffusion_trainingset/valid/real_coco.txt'
keep_dir = '/nobackup3/anirudh/datasets/coco/val/0_real'
move_dir = '/nobackup3/anirudh/datasets/coco/val/0_real_rem'

process_images(image_dir, text_file, keep_dir, move_dir)

