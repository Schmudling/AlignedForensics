IMAGES_PATH="/nobackup3/anirudh/datasets/coco/train/0_real"
SHADERS_TRAIN="/nobackup2/anirudh/datasets/Shaders21k/shaders_inv_png/train/0_real"

#CUDA_VISIBLE_DEVICES=0 python execute.py --input_folder=$IMAGES_PATH --batch_size 32 --repo_id "playgroundai/playground-v2.5-1024px-aesthetic" 

CUDA_VISIBLE_DEVICES=0 python execute.py --input_folder=$IMAGES_PATH --batch_size 1 --repo_id 'stabilityai/stable-diffusion-3-medium-diffusers' 
#CUDA_VISIBLE_DEVICES=0 python execute.py --input_folder=$SHADERS_TRAIN --batch_size 64 --repo_id 'runwayml/stable-diffusion-v1-5' 

#--use_ddim --steps=100
