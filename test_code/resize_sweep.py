
import torch
import os
import pandas
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.processing import make_normalize
import matplotlib.pyplot as plt
import tqdm
import glob
import sys
import yaml
import json
import io
from PIL import Image
import seaborn as sns
from custom_transforms import DiagonalShift
from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights
from dataset import UnlabeledImageDataset
import torchvision.transforms as transforms
import statistics
from sweepers import *

def plot_means(models_dict, x_values, x_label, y_label, title, filename):
    # Rename models for readability
    if "Corvi23_170k_inv_ldm" in models_dict:
        models_dict["Ours"] = models_dict.pop("Corvi23_170k_inv_ldm")
    if 'Corvi23_170k_inv_ldm_sync' in models_dict:
        models_dict["Ours-sync"] = models_dict.pop("Corvi23_170k_inv_ldm_sync")
    if 'Corvi23_90k_inv_ldm' in models_dict:
        models_dict["Ours (only COCO)"] = models_dict.pop("Corvi23_90k_inv_ldm")

    plt.figure(figsize=(10, 6))

    for model_name, stats in models_dict.items():
        # Plot mean line
        plt.plot(x_values, stats['mean'], label=model_name)

        # Plot mean ± standard deviation as a shaded area
        if 'sd' in stats:
            mean = np.array(stats['mean'])
            sd = np.array(stats['sd'])
            plt.fill_between(
                x_values, mean - sd, mean + sd,
                color=plt.gca().lines[-1].get_color(), alpha=0.2,
                label=f'{model_name} ±1 SD'
            )

    # Set plot properties
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save and close plot
    plt.savefig(filename)
    plt.close()




def save_confusion_matrix_heatmaps(confusion_dict,save_dir):
    if "Corvi23_170k_inv_ldm" in confusion_dict.keys():
        confusion_dict["Ours"] = confusion_dict.pop("Corvi23_170k_inv_ldm")
    if 'Corvi23_170k_inv_ldm_sync' in confusion_dict.keys():
        confusion_dict["Ours-sync"] = confusion_dict.pop("Corvi23_170k_inv_ldm_sync")
    if 'Corvi23_90k_inv_ldm' in confusion_dict.keys():
        confusion_dict["Ours (only COCO)"] = confusion_dict.pop("Corvi23_90k_inv_ldm")
    plt.figure(figsize=(10, 6))


    for model_name, matrix in confusion_dict.items():
        # Replace slashes in model_name to make it a valid file name
        file_name = os.path.join(save_dir, model_name.replace('/', '_') + '.png')
        
        # Extracting sizes from matrix keys, splitting them into true/predicted dimensions
        sizes = [tuple(map(int, size.split('*'))) for size in matrix.keys()]
        unique_true_sizes = sorted(set(size[0] for size in sizes))
        unique_pred_sizes = sorted(set(size[1] for size in sizes))

        # Initialize an empty array to store the accuracy values
        matrix_array = np.zeros((len(unique_true_sizes), len(unique_pred_sizes)))
        
        # Fill the matrix array with accuracy values from the dictionary
        for tidx,true_size in enumerate(unique_true_sizes):
            for pidx, pred_size in enumerate(unique_pred_sizes):
                matrix_array[tidx][pidx] = confusion_dict[model_name][f'{true_size}*{pred_size}']
        
        # Create the heatmap using seaborn
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix_array, annot=False, fmt=".3f", cmap='magma', 
                    xticklabels=unique_pred_sizes, yticklabels=unique_true_sizes)
        plt.title(model_name)
        plt.xlabel('Fake Resolution')  # Set x-axis label
        plt.ylabel('Real Resolution')  # Set y-axis label

        # Save the heatmap as an image
        plt.savefig(file_name)
        plt.close()
        
        print(f"Saved heatmap for {model_name} to {file_name}")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, help="The real folder")
    parser.add_argument("--fake", type=str, help="The fake folder")
    parser.add_argument("--out_path", type=str, help="Path to save the plot", default='plots')
    parser.add_argument("--setting", type=str, help="Enter either res or qual, based on whether to sweep resolution or quality", default='res')
    parser.add_argument("--toggle", type=str, help="toggle to get an accuracy matrix", default=None)
    parser.add_argument("--weights_dir", type=str, help="The directory to the networks weights", default="./weights/")
    parser.add_argument("--models", type=str, help="List of models to test", default='post_iclr/lp/inv_sync_ldm,post_iclr/icml/inv_sync_ldm_staypos_relu,post_iclr/icml/inv_sync_ldm_staypos_relu_lay4')
    parser.add_argument("--device", type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--start", type=float, help="Starting setting", default=256)
    parser.add_argument("--end", type=float, help="Final setting", default=1024)
    parser.add_argument("--stride", type=int, help="Stride", default=2)
    parser.add_argument("--base_res", type=int, help="ogres", default=512)
    parser.add_argument("--bandwidth", type=int, help="Bandwidth of frequencies to be retained, like 0-20, 10-30 etc", default=None)
    parser.add_argument('--drop_single', action='store_true', help='Setting this to true will only drop a specifc bit in the image')
    parser.add_argument('--use_proj', action='store_true', help='Use a projection layer, before contrastive training')
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    if args['setting'] == 'res':
        if args['toggle'] == 'confusion_matrix':
            accs = resolutions_confusion_matrix(models_list=args['models'],real_folder=args['real'], fake_folder=args['fake'], start_res=int(args['start']), end_res=int(args['end']), step=args['stride'],args=args)
            print(accs)
            save_confusion_matrix_heatmaps(accs, args['out_path'])
            sys.exit("Ending the program abruptly.")

        real_probs, fake_probs = load_datasets_with_resolutions(models_list=args['models'],real_folder=args['real'], fake_folder=args['fake'], start_res=int(args['start']), end_res=int(args['end']), step=args['stride'], args=args)
        with open('real_swp.json', 'w') as json_file:
            json.dump(real_probs, json_file, indent=4)
        with open('fake_swp.json', 'w') as json_file:
            json.dump(fake_probs, json_file, indent=4)
        x = create_scale_list(start_res=int(args['start']), end_res=int(args['end']), step=args['stride'])
        sweeper = 'Scaling Factor'
    elif args['setting'] == 'qual':
        real_probs, fake_probs = load_datasets_with_compression(models_list=args['models'], real_folder=args['real'], fake_folder=args['fake'], start_quality=int(args['start']), end_quality=int(args['end']), step=args['stride'], args=args)
        #x = create_scale_list(start_res=int(args['start']), end_res=int(args['end']), step=args['stride'], isQual=True)
        with open('real_swp_webp.json', 'w') as json_file:
            json.dump(real_probs, json_file, indent=4)
        with open('fake_swp_webp.json', 'w') as json_file:
            json.dump(fake_probs, json_file, indent=4)
        x = list(range(int(args['start']), int(args['end'])+1, args['stride']))
        sweeper = 'Webp compression quality'
    elif args['setting'] == 'freq':
        real_probs, fake_probs = load_datasets_with_frequency_sweep(models_list=args['models'], real_folder=args['real'], fake_folder=args['fake'], start_percent=int(args['start']), end_percent=int(args['end']), step=args['stride'], args=args)
        x = list(range(int(args['start']), int(args['end'])+1, args['stride']))
        #x = create_scale_list(start_res=args['start_quality'], end_res=args['end_quality'], step=args['stride'])
    elif args['setting'] == 'quant':
        real_probs, fake_probs = load_datasets_with_quantization(models_list=args['models'], real_folder=args['real'], fake_folder=args['fake'], start_bits=int(args['start']), end_bits=int(args['end']), step=args['stride'], args=args)
        x = list(range(int(args['start']), int(args['end'])+1, args['stride']))
    x = [t * (args['start']/args['base_res']) for t in x]
    if args['setting'] == 'res':
        plot_means(models_dict=real_probs, x_values=x, x_label='Scaling Factor', y_label='Probability of Image Being Fake', title='Real Images', filename=os.path.join(args["out_path"],'real_means.png'))
        plot_means(models_dict=fake_probs, x_values=x, x_label='Scaling Factor', y_label='Probability of Image Being Fake', title='Fake Images', filename=os.path.join(args["out_path"],'fake_means.png'))
    elif args['setting'] == 'qual':
        x = list(range(int(args['start']), int(args['end'])+1, int(args['stride'])))        
        plot_means(models_dict=real_probs, x_values=x, x_label='Compression Quality', y_label='Probability of Image Being Fake', title='Real Images', filename=os.path.join(args["out_path"],'real_means.png'))
        plot_means(models_dict=fake_probs, x_values=x, x_label='Compression Quality', y_label='Probability of Image Being Fake', title='Fake Images', filename=os.path.join(args["out_path"],'fake_means.png'))

    for key in args['models']:
        #key = key.split('/')[-1]
        path = os.path.join(args["out_path"],key+'.png')
        plot_detailed_spread(real_stats=real_probs[key], fake_stats=fake_probs[key], filename=os.path.join(args["out_path"],key.split('/')[-1]+'.png'), x=x, sweeper=sweeper)
        #plot_mean_std(real_means=real_probs[key]['means'], real_std=real_probs[key]['sd'], fake_means=fake_probs[key]['means'], fake_std=fake_probs[key]['sd'], filename=path,x=x,sweeper=sweeper)
