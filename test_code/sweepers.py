import torch
import os
import pandas
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.processing import make_normalize
import matplotlib.pyplot as plt
import tqdm
import math
import glob
import sys
import yaml
import io
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights
from dataset import UnlabeledImageDataset
import torchvision.transforms as transforms
import statistics

def create_scale_list(start_res, end_res, step, isQual=False):
    num_steps = (end_res - start_res) // step + 1
    if isQual:
        scale_list = [start_res + i*step for i in range(num_steps)]
    else:
        scale_list = [(start_res + i*step)/start_res for i in range(num_steps)]
    return scale_list

def plot_mean_std(real_means, real_std, fake_means, fake_std, filename='mean_std_plot.png', x=None, sweeper='Scaling Factor'):
        plt.figure(figsize=(10, 6))

        real_lower = np.clip(np.array(real_means) - 0*np.array(real_std), 0, 1)
        real_upper = np.clip(np.array(real_means) + 0*np.array(real_std), 0, 1)
        fake_lower = np.clip(np.array(fake_means) - 0*np.array(fake_std), 0, 1)
        fake_upper = np.clip(np.array(fake_means) + 0*np.array(fake_std), 0, 1)

        plt.plot(x, real_means, label='Real Images', color='green')
        plt.fill_between(x, real_lower, real_upper, color='green', alpha=0.2)

        plt.plot(x, fake_means, label='Fake Images', color='red')
        plt.fill_between(x, fake_lower, fake_upper, color='red', alpha=0.2)

        plt.xlabel(sweeper, fontsize=12)
        plt.ylabel("Probability of Image Being Fake", fontsize=14, fontweight='bold')
        #plt.ylabel('Prob(fake)')
        #plt.title('')
        plt.legend()
        plt.grid(True)

        plt.savefig(filename)
        plt.close()


def plot_detailed_spread(real_stats, fake_stats, filename='detailed_spread_plot.png', x=None, sweeper='Scaling Factor'):
    plt.figure(figsize=(10, 6))

    # Define color mappings for the different components
    stat_colors = {
        'mean': ('green', 'red'),
        'iqr': (0.15, 0.15),
        'sd_line': ('green', 'red'),
        'percentile_line': ('blue', 'purple')
    }

    # Plot for real images
    plt.plot(x, real_stats['mean'], label='Real Images', color='green')

    # Standard deviation range with dotted lines around the mean
    if 'sd' in real_stats:
        real_lower_sd = np.clip(np.array(real_stats['mean']) - np.array(real_stats['sd']), 0, 1)
        real_upper_sd = np.clip(np.array(real_stats['mean']) + np.array(real_stats['sd']), 0, 1)
        plt.plot(x, real_lower_sd, color=stat_colors['sd_line'][0], linestyle='dotted', label='Real Std Dev Lower')
        plt.plot(x, real_upper_sd, color=stat_colors['sd_line'][0], linestyle='dotted', label='Real Std Dev Upper')

    # Interquartile range with filled color
    if '25th_percentile' in real_stats and '75th_percentile' in real_stats:
        real_lower_iqr = np.clip(real_stats['25th_percentile'], 0, 1)
        real_upper_iqr = np.clip(real_stats['75th_percentile'], 0, 1)
        plt.fill_between(x, real_lower_iqr, real_upper_iqr, color='green', alpha=stat_colors['iqr'][0], label='Real IQR')

    # 5th and 95th percentiles with dashed lines
    if '5th_percentile' in real_stats and '95th_percentile' in real_stats:
        real_lower_p5 = np.clip(real_stats['5th_percentile'], 0, 1)
        real_upper_p95 = np.clip(real_stats['95th_percentile'], 0, 1)
        plt.plot(x, real_lower_p5, color=stat_colors['percentile_line'][0], linestyle='--', label='Real 5th Percentile')
        plt.plot(x, real_upper_p95, color=stat_colors['percentile_line'][0], linestyle='--', label='Real 95th Percentile')

    # Plot for fake images
    plt.plot(x, fake_stats['mean'], label='Fake Images', color='red')

    # Standard deviation range with dotted lines around the mean for fake images
    if 'sd' in fake_stats:
        fake_lower_sd = np.clip(np.array(fake_stats['mean']) - np.array(fake_stats['sd']), 0, 1)
        fake_upper_sd = np.clip(np.array(fake_stats['mean']) + np.array(fake_stats['sd']), 0, 1)
        plt.plot(x, fake_lower_sd, color=stat_colors['sd_line'][1], linestyle='dotted', label='Fake Std Dev Lower')
        plt.plot(x, fake_upper_sd, color=stat_colors['sd_line'][1], linestyle='dotted', label='Fake Std Dev Upper')

    # Interquartile range with filled color for fake images
    if '25th_percentile' in fake_stats and '75th_percentile' in fake_stats:
        fake_lower_iqr = np.clip(fake_stats['25th_percentile'], 0, 1)
        fake_upper_iqr = np.clip(fake_stats['75th_percentile'], 0, 1)
        plt.fill_between(x, fake_lower_iqr, fake_upper_iqr, color='red', alpha=stat_colors['iqr'][1], label='Fake IQR')

    # 5th and 95th percentiles with dashed lines for fake images
    if '5th_percentile' in fake_stats and '95th_percentile' in fake_stats:
        fake_lower_p5 = np.clip(fake_stats['5th_percentile'], 0, 1)
        fake_upper_p95 = np.clip(fake_stats['95th_percentile'], 0, 1)
        plt.plot(x, fake_lower_p5, color=stat_colors['percentile_line'][1], linestyle='--', label='Fake 5th Percentile')
        plt.plot(x, fake_upper_p95, color=stat_colors['percentile_line'][1], linestyle='--', label='Fake 95th Percentile')

    plt.xlabel(sweeper, fontsize=12)
    plt.ylabel("Probability of Image Being Fake", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()


def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    if 'stay_positive' in data.keys():
        stay_positive = data['stay_positive']
    else:
        stay_positive = None
    if 'layer' in data.keys():
        layer = data['layer'].split(',')
    else:
        layer = None
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size'], stay_positive, layer



def get_models(weights_dir, models_list, device, use_proj=False):
    norm_type = 'resnet'
    models_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size, stay_positive, layer = get_config(model_name, weights_dir=weights_dir)
        print("SIZE:", patch_size)
        

        if 'proj' in model_name:
            model = load_weights(create_architecture(arch,use_proj=True, scale_factor=4), model_path)
        else:
            model = load_weights(create_architecture(arch, stay_positive=stay_positive, aux_layers=layer), model_path)


        #model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()
        models_dict[model_name] = model
        print(flush=True)
    return models_dict


def run_models(models_dict, loader,device,raw=False):
    probs = {}
    for images in loader:
        for name in models_dict.keys():
            if name not in probs:
                probs[name] = []

            scores = models_dict[name](images.to(device)).cpu()
            probs[name].extend(torch.sigmoid(scores).tolist())
    for key in probs:

        p = np.array(probs[key]).flatten().tolist()
        if raw:
            probs[key] = p
        else:
            probs[key] = (statistics.mean(p), statistics.stdev(p))
    return probs

def run_models_adv(models_dict, loader, device, raw=False):
    probs = {}
    for images in loader:
        for name in models_dict.keys():
            if name not in probs:
                probs[name] = []

            scores = models_dict[name](images.to(device)).cpu()
            probs[name].extend(torch.sigmoid(scores).tolist())
    
    for key in probs:
        p = np.array(probs[key]).flatten()
        if raw:
            probs[key] = p.tolist()
        else:
            # Compute required spread metrics
            mean = statistics.mean(p)
            std_dev = statistics.stdev(p)
            p25 = np.percentile(p, 25)
            p75 = np.percentile(p, 75)
            iqr = p75 - p25
            p5, p95 = np.percentile(p, 5), np.percentile(p, 95)

            # Store metrics in a dictionary
            probs[key] = {
                "mean": mean,
                "sd": std_dev,
                "iqr": iqr,
                "25th_percentile": p25,
                "75th_percentile": p75,
                "5th_percentile": p5,
                "95th_percentile": p95
            }
    return probs


def apply_low_pass_filter(image, percent, bandwidth=None):
    rows, cols = image.shape[:2]
    radius = int(min(rows, cols)// math.sqrt(2) * percent / 100)
    print(radius)
    crow, ccol = rows // 2, cols // 2  # center

    # Create a mask with a circular low-pass filter
    mask = np.zeros((rows, cols), np.uint8)
    if bandwidth is None:
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    else:
        low_radius = int(min(rows, cols) * percent / 100)
        high_radius = int(min(rows, cols) * (percent+bandwidth) / 100)
        cv2.circle(mask, (ccol, crow), high_radius, 1, thickness=-1)
        cv2.circle(mask, (ccol, crow), low_radius, 0, thickness=-1)

    # Apply the mask to each channel
    channels = cv2.split(image)
    filtered_channels = []
    for channel in channels:
        fshift = np.fft.fftshift(np.fft.fft2(channel))
        #CHANGE BACK
        fshift_masked = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_masked)
        img_back = np.fft.ifft2(f_ishift)
        print(img_back)
        img_back = np.abs(img_back).astype(np.float32)
        filtered_channels.append(img_back)
    # Merge the filtered channels back into a color image
    #filtered_image = cv2.merge(channels)
    filtered_image = cv2.merge(filtered_channels)
    return Image.fromarray(np.uint8(filtered_image))


def load_datasets_with_frequency_sweep(models_list, real_folder, fake_folder, start_percent, end_percent, step, args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'])
    real_probs = {}
    fake_probs = {}

    for key in models_dict.keys():
        real_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [], "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
        fake_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [], "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
    
    for quality in range(start_quality, end_quality + 1, step):
        print(f"Loading datasets with WebP compression quality {quality}")

        transform = transforms.Compose([
            transforms.Lambda(low_pass_transform),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        real_dataset = UnlabeledImageDataset(real_folder, transform=transform)
        fake_dataset = UnlabeledImageDataset(fake_folder, transform=transform)

        # Adjust batch size according to the quality if necessary
        bs = 20
        # bs = 80 if quality < 50 else 30 if quality < 75 else 10

        real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True)
        fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=True)

        with torch.no_grad():
            real_stats = run_models_adv(models_dict, real_loader, device=args['device'])
            fake_stats = run_models_adv(models_dict, fake_loader, device=args['device'])
        #print(real_stats)
        print(fake_stats)

        for key in models_dict.keys():
            for key_in in real_stats[key].keys():
                real_probs[key][key_in].append(real_stats[key][key_in])
                fake_probs[key][key_in].append(fake_stats[key][key_in])
            #real_probs[key]['means'].append(real_stats[key][0])
            #real_probs[key]['sd'].append(real_stats[key][1])
            #fake_probs[key]['means'].append(fake_stats[key][0])
            #fake_probs[key]['sd'].append(fake_stats[key][1])
    print(quality, range(start_quality, end_quality + 1, step))
    return real_probs, fake_probs


def webp_compress(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    return Image.open(buffer)

def load_datasets_with_compression(models_list, real_folder, fake_folder, start_quality, end_quality, step, args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'])
    real_probs = {}
    fake_probs = {}

    for key in models_dict.keys():
        real_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [], "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
        fake_probs[key] = {'mean': [], 'sd': [], "iqr": [], "25th_percentile": [], "5th_percentile": [], "75th_percentile": [], "95th_percentile": []}
    
    for quality in range(start_quality, end_quality + 1, step):
        print(f"Loading datasets with WebP compression quality {quality}")

        transform = transforms.Compose([
            transforms.Lambda(lambda img: webp_compress(img, quality)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        real_dataset = UnlabeledImageDataset(real_folder, transform=transform)
        fake_dataset = UnlabeledImageDataset(fake_folder, transform=transform)

        # Adjust batch size according to the quality if necessary
        bs = 20
        # bs = 80 if quality < 50 else 30 if quality < 75 else 10

        real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True)
        fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=True)

        with torch.no_grad():
            real_stats = run_models_adv(models_dict, real_loader, device=args['device'])
            fake_stats = run_models_adv(models_dict, fake_loader, device=args['device'])
        #print(real_stats)
        print(fake_stats)

        for key in models_dict.keys():
            for key_in in real_stats[key].keys():
                real_probs[key][key_in].append(real_stats[key][key_in])
                fake_probs[key][key_in].append(fake_stats[key][key_in])
            #real_probs[key]['means'].append(real_stats[key][0])
            #real_probs[key]['sd'].append(real_stats[key][1])
            #fake_probs[key]['means'].append(fake_stats[key][0])
            #fake_probs[key]['sd'].append(fake_stats[key][1])
    print(quality, range(start_quality, end_quality + 1, step))
    return real_probs, fake_probs


def drop_least_significant_bits(image, bits_to_drop, drop_single=False):
    # Create a mask to drop the least significant bits
    mask = 255 << bits_to_drop
    if drop_single:
        mask = ~(1 << bits_to_drop)
        #new_number = number & mask
        #mask = ~(1 << bits_to_drop)
    # Apply the mask to each channel of the image
    quantized_image = np.bitwise_and(np.array(image), mask).astype(np.uint8)
    return Image.fromarray(quantized_image)

def load_datasets_with_quantization(models_list, real_folder, fake_folder, start_bits, end_bits, step, args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'])
    real_probs = {}
    fake_probs = {}
    for key in models_dict.keys():
        real_probs[key] = {}
        fake_probs[key] = {}
        real_probs[key]['means']=[]
        fake_probs[key]['means']=[]
        real_probs[key]['sd']=[]
        fake_probs[key]['sd']=[]
    
    for bits_to_drop in range(start_bits, end_bits + 1, step):
        print(f"Loading datasets with {bits_to_drop} bits dropped")
        
        def quant_transform(image, bits_to_drop, drop_single=False):
            return drop_least_significant_bits(image, bits_to_drop,drop_single=drop_single)
        
        transform = transforms.Compose([
            transforms.Lambda(lambda img: quant_transform(img, bits_to_drop,drop_single=args['drop_single'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        real_dataset = UnlabeledImageDataset(real_folder, transform=transform)
        fake_dataset = UnlabeledImageDataset(fake_folder, transform=transform)
        
        bs = 50
        #CHANGE BACK TO batch_size=bs
        real_loader = DataLoader(real_dataset, batch_size=50, shuffle=True)
        fake_loader = DataLoader(fake_dataset, batch_size=20, shuffle=True)
        
        with torch.no_grad():
            real_stats = run_models(models_dict, real_loader, device=args['device'])
            fake_stats = run_models(models_dict, fake_loader, device=args['device'])
        
        print(fake_stats)
        for key in models_dict.keys():
            real_probs[key]['means'].append(real_stats[key][0])
            real_probs[key]['sd'].append(real_stats[key][1])
            fake_probs[key]['means'].append(fake_stats[key][0])
            fake_probs[key]['sd'].append(fake_stats[key][1])
    return real_probs, fake_probs



def create_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC, antialias=True),  # Resize images to the specified resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#RESNET NORM
    ])


def load_datasets_with_resolutions(models_list,real_folder, fake_folder, start_res, end_res, step,args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'], use_proj=args['use_proj'])
    real_probs = {}
    fake_probs = {}
    for key in models_dict.keys():
        real_probs[key] = {}
        fake_probs[key] = {}
        real_probs[key]['means']=[]
        fake_probs[key]['means']=[]
        real_probs[key]['sd']=[]
        fake_probs[key]['sd']=[]
    for res in range(start_res, end_res + 1, step):
        print(f"Loading datasets with resolution {res}x{res}")
        transform = create_transform(res)

        real_dataset = UnlabeledImageDataset(real_folder, transform=transform)
        fake_dataset = UnlabeledImageDataset(fake_folder, transform=transform)
        if res <300:
            #120
            bs=80
        elif res<500:
            #50
            bs=30
        elif res< 1000:
            bs = 10
        else:
            bs=10
        real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True)
        fake_loader = DataLoader(fake_dataset, batch_size=bs, shuffle=True)
        with torch.no_grad():
            real_stats = run_models(models_dict, real_loader,device=args['device'])
            fake_stats = run_models(models_dict, fake_loader,device=args['device'])
        #print(fake_stats)
        for key in models_dict.keys():
            real_probs[key]['means'].append(real_stats[key][0])
            real_probs[key]['sd'].append(real_stats[key][1])
            fake_probs[key]['means'].append(fake_stats[key][0])
            fake_probs[key]['sd'].append(fake_stats[key][1])
    return real_probs, fake_probs



def resolutions_confusion_matrix(models_list,real_folder, fake_folder, start_res, end_res, step,args):
    models_dict = get_models(weights_dir=args['weights_dir'], models_list=models_list, device=args['device'], use_proj=args['use_proj'])
    accs = {}
    for key in models_dict.keys():
        accs[key] = {}
    for real_res in range(start_res, end_res + 1, step):
        for fake_res in range(start_res, end_res + 1, step):
            print(f"Loading datasets with resolution {real_res}x{fake_res}")
            real_transform = create_transform(real_res)
            fake_transform = create_transform(fake_res)
            real_dataset = UnlabeledImageDataset(real_folder, transform=real_transform)
            fake_dataset = UnlabeledImageDataset(fake_folder, transform=fake_transform)
            if real_res <300:
            #120
                rbs=80
            if fake_res <300:
                fbs=80
            if real_res<500:
                rbs=30
            if fake_res<500:
                fbs=30
            rbs=10
            fbs=10
            real_loader = DataLoader(real_dataset, batch_size=rbs, shuffle=True)
            fake_loader = DataLoader(fake_dataset, batch_size=fbs, shuffle=True)
            with torch.no_grad():
                real_stats = run_models(models_dict, real_loader,device=args['device'],raw=True)
                fake_stats = run_models(models_dict, fake_loader,device=args['device'],raw=True)
            for key in models_dict.keys():
                r_acc = len([p for p in real_stats[key] if p < 0.5])/len(real_stats[key])
                f_acc = len([p for p in fake_stats[key] if p >= 0.5])/len(fake_stats[key])
                acc = (r_acc + f_acc)/2
                config = str(real_res) + "*" + str(fake_res)
                accs[key][config] = acc
                #real_probs[key][][].append(real_stats[key][0])
                #real_probs[key]['sd'].append(real_stats[key][1])
                #fake_probs[key]['means'].append(fake_stats[key][0])
                #fake_probs[key]['sd'].append(fake_stats[key][1])
    print(accs)
    return accs
