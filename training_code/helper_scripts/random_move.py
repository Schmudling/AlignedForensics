import os
import random
import shutil

def move_random_files(source_dir, dest_dir, num_files):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get a list of all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # If the number of files to move is more than the available files, limit it
    if num_files > len(all_files):
        print(f"Only {len(all_files)} files available, moving all.")
        num_files = len(all_files)
    
    # Select a random sample of files to move
    files_to_move = random.sample(all_files, num_files)
    
    # Move each selected file to the destination directory
    for file_name in files_to_move:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
        print(f"Moved {file_name} to {dest_dir}")

# Example usage
source_dir = '/nobackup3/anirudh/DMimageDetection/training_code/latent_diffusion_trainingset/train/0_real/'
dest_dir = '/nobackup3/anirudh/DMimageDetection/training_code/latent_diffusion_trainingset/valid/0_real/'
num_files = 30000

move_random_files(source_dir, dest_dir, num_files)

