from utils.tools import load_data
import torch
from models.ours import ours_nn
from tqdm import tqdm
from utils.LossIndex import com_metrics
import os
from os.path import join
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter


def normalization(data):
    data = data - np.mean(data)
    data = data / np.std(data)
    return data

def syn_nn_result(seis,net):
    input_block = torch.from_numpy(seis[None,None, ...]).cuda().float()
    sample_output,_ = net(input_block)  
    sample_output = sample_output.squeeze().detach().cpu().numpy()  
    return sample_output

def sliding_window_prediction_3d(input_data, model, block_size, device):
    overlap = 0.5
    input_shape = input_data.shape
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.cpu().numpy()  
        input_shape = input_data.shape
    if len(input_shape) != 4:
        raise ValueError(f"Expected input_data shape (C, D, H, W), but got {input_shape}")
    C, D, H, W = input_shape

    block_shape = np.array(block_size)  
    step = (1 - overlap) * block_shape 

    num_blocks = np.maximum(np.ceil((np.array([D, H, W]) - block_shape) / step).astype(int) + 1, 1)
    sliding_shape = ((num_blocks - 1) * step + block_shape).astype(int)
    sliding_data = np.zeros((C, *sliding_shape), dtype=input_data.dtype)
    sliding_data[:, :D, :H, :W] = input_data

    output = np.zeros((1, *sliding_shape), dtype=np.float32)
    weight_map = np.zeros(sliding_shape, dtype=np.float32)

    total_iterations = np.prod(num_blocks)
    progress_bar = tqdm(total=total_iterations, desc='[Pred]', unit='it')
    with torch.no_grad():
        for i in range(num_blocks[0]):
            for j in range(num_blocks[1]):
                for k in range(num_blocks[2]):
                    start = (step * np.array([i, j, k])).astype(int)  
                    end = (start + block_shape).astype(int)
                    block = sliding_data[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]  
                    
                    input_block = torch.from_numpy(block[None, ...]).to(device).float()

                    block_prediction,_ = model(input_block)

                    block_prediction = block_prediction.cpu().numpy().squeeze(0)
                    output[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] += block_prediction
                    weight_map[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += 1  

                    progress_bar.update(1)
    progress_bar.close()
    output /= np.expand_dims(weight_map, axis=0)  
    smoothed_output = gaussian_filter(output, sigma=(0, 0.0, 0.0, 0.0))  
    out = smoothed_output[:, :D, :H, :W]
    out = out.squeeze(0)
    return out


def pred(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---")
    print('Device is :', device)  

    save_path = join('./EXP/', args.exp, 'result')  
    os.makedirs(save_path, exist_ok=True) 

    print("Loading Model ... ")
    if args.exp == 'ours':
        model = ours_nn().to(device)

    checkpoint = torch.load(args.pretrained_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = checkpoint['net']
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("Detected DataParallel model, removing 'module.' prefix...")
        state_dict = {k.replace("module.", "",1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if args.pre_data == 'syn':
        seis = np.fromfile(args.pre_path, dtype=np.single)
        seis = seis.reshape(tuple([128]*3))
        seis = normalization(seis)
        output = syn_nn_result(seis,model)
    elif args.pre_data == 'field':
        field_data = np.load(args.pre_path)
        data = normalization(field_data)
        block_size = (128, 128, 128)        
        input_data = torch.from_numpy(data[None, ...]) 
        output = sliding_window_prediction_3d(input_data, model,(128, 128, 128), device)

    np.save(join(save_path, f'{args.pre_data}.npy'), output)







def data_to(inputs,labels):
    device = torch.device('cuda')
    return inputs.float().to(device), labels.float().to(device)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---")
    print('Device is :', device)

    # save_path
    save_path = join('./EXP/', args.exp) 

    # Load data
    print("---")
    print("Loading data ... ")
    val_loader = load_data(args)
    print(val_loader)

    print("Loading Model ... ")
    if args.exp == 'ours':
        model = ours_nn().to(device)

    checkpoint = torch.load(args.pretrained_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = checkpoint['net']
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("Detected DataParallel model, removing 'module.' prefix...")
        state_dict = {k.replace("module.", "",1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    with torch.no_grad():
        val_metrics = {k: 0.0 for k in ["Precision", "Recall","F1-score", "IoU", "mIoU","HD","HD95"]}
        for step, (inputs,labels,filenames) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, labels = data_to(inputs,labels)
            outputs,lowout = model(inputs)
            step_metrics = com_metrics(outputs, labels)
            for key, value in step_metrics.items():
                val_metrics[key] += value
        avg_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
        print(" ".join(f"val {k}: {v :.4f}" for k, v in avg_metrics.items()))
        save_file = os.path.join(save_path, 'ours.xlsx')
        df = pd.DataFrame([avg_metrics.values()], columns=avg_metrics.keys())
        df.to_excel(save_file, index=False)
        print("Finished ! ")