import torch
import numpy as np
import os
import cv2
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from dataloader.dataloader import FaultDataset3d



''
def show_result(input,output,save_path,num,dim,pos,args):
    input_normalized= cv2.normalize(input, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    input_colored = np.stack([input_normalized]*3, axis=-1)
    
    cmap = LinearSegmentedColormap.from_list('transparent_jet', [(0, (1, 1, 1, 0)),  # 小于 0.4 透明
                    (args.threshold, 'blue'),(1, 'red')])
                    # (0.4, 'blue'),(0.6, 'cyan'),(0.8, 'yellow'),(1, 'red')])

    color_map = cmap(output)

    # height, width = output.shape
    # color_map = np.zeros((height, width, 4))
    # for i in range(height):
    #     for j in range(width):
    #         prob = output[i, j]
    #         normalized_value = norm(prob)
    #         color_map[i, j] = cmap(normalized_value)
            
    color_pre = (color_map[:, :, :3] * 255).astype(np.uint8)
    color_pre = cv2.cvtColor(color_pre, cv2.COLOR_RGB2BGR)  # 转换为 BGR
    overlay = cv2.addWeighted(input_colored,  1, color_pre, 0.2, 0)
    cv2.imwrite(save_path + str(num)+'_dim'+str(dim) +'_'+str(pos)+'.png',overlay)
    

def sliding_window_prediction_3d(input_data, model, block_size, args):
    overlap = args.overlap
    input_shape = input_data.shape
    block_shape = np.array(block_size)  # 128*128*128  # 切块大小和步长
    step = (1 - overlap) * block_shape  # 64*64*64
    num_blocks = np.maximum(np.ceil((input_shape - block_shape) / step).astype(int) + 1, 1)
    # num_blocks = np.ceil(input_shape / step).astype(int)   # 计算需要切割成的块数 # ceil先上取整,floor向下取整
    # 初始化预测结果和权重矩阵
    sliding_shape = ((num_blocks - 1) * step + block_shape).astype(int)

    sliding_data = np.zeros(sliding_shape)
    sliding_data[0:input_shape[0], 0:input_shape[1], 0:input_shape[2]] = input_data
    output = np.zeros(sliding_shape)
    weight_map = np.zeros(sliding_shape)
    total_iterations = np.prod(num_blocks)
    # total_iterations = num_blocks[0] * num_blocks[1] * num_blocks[2]
    progress_bar = tqdm(total=total_iterations, desc='[Pred]', unit='it')
    # 滑动窗口切块和预测
    with torch.no_grad():
        for i in range(num_blocks[0]):
            for j in range(num_blocks[1]):
                for k in range(num_blocks[2]):
                    start = (step * np.array([i, j, k])).astype(int)  # 计算当前块的起始和结束位置
                    end = (start + block_shape).astype(int)
                    block = sliding_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]  #裁剪当前块的数据
                    
                    # block = block.reshape((1, 1, block.shape[0], block.shape[1], block.shape[2]))
                    # input_block = torch.from_numpy(block).to(args.device).float()
                    input_block = torch.from_numpy(block[None, None]).to(args.device).float()
                    # block_prediction = model(input_block)
                    # block_prediction = block_prediction.detach().cpu().numpy()
                    # block_prediction = np.squeeze(block_prediction)
                    block_prediction,lowout = model(input_block)
                    block_prediction = block_prediction.cpu().numpy().squeeze()
                    weight_map[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += 1  
                    # 计算当前块的权重矩阵
                    output[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += block_prediction  
                    # 将当前块的预测结果叠加到输出中
                    progress_bar.update(1)
    progress_bar.close()
    output /= weight_map  # 根据权重矩阵对预测结果进行归一化
    smoothed_output = gaussian_filter(output, sigma=args.sigma)  # 使用高斯滤波器对边界进行平滑
    out = smoothed_output[0:input_shape[0], 0:input_shape[1], 0:input_shape[2]]
    # h,w,d = out.shape
    # threshold = 0.4
    out[out <= args.threshold] = 0   # 这里都保留了0-1的值,后面画图可以改, 有影响

    return out


'合成数据可视化'
def visual_result_3d(args,net,num,kind):
    net.eval()
    with torch.no_grad():
        if kind == 'valid':
            dataset = FaultDataset3d(args.valid_path, args.data_auge,'valid')
        else: 
            dataset = FaultDataset3d(args.train_path, args.data_auge,'train')

        sample_indices = np.random.choice(len(dataset), size=1, replace=False)

        for idx in sample_indices:
            sample_data, sample_target,filename = dataset[idx]  # [C,128,128,128]

            sample_data = sample_data.unsqueeze(0).float().cuda()  # [B,C,128,128,128]
            # print(sample_data.shape)
            sample_output, lowout = net(sample_data)  # torch.Size([1, 1, 128, 128, 128])   
            sample_output = sample_output.squeeze().detach().cpu().numpy()  # (128, 128, 128)
            sample_data = sample_data.squeeze().detach().cpu().numpy()
            sample_target = sample_target.squeeze().detach().cpu().numpy()
            lowout = lowout.squeeze().detach().cpu().numpy()
            result_path = './EXP/' + args.exp + '/results/'+kind+'/'  # +str(num)+'/'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            canvas = np.zeros((150, 567))
            data_normalized = cv2.normalize(np.transpose(sample_data[64,:,:]), None, 0, 255, cv2.NORM_MINMAX)
            canvas[11:139, 11:139] = data_normalized
            cv2.putText(canvas, 'input', (75, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25,255,1,cv2.LINE_AA)
            sample_output = sample_output[64,:,:]
            # sample_output[sample_output <= args.threshold] = 0   
            image2 = np.transpose(sample_output*255)
            canvas[11:139, 150:278] = image2
            cv2.putText(canvas, 'output', (214, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25,255,1,cv2.LINE_AA)
            sample_target=sample_target[64,:,:]
            image3 = np.transpose(sample_target*255)
            canvas[11:139, 289:417] = image3
            cv2.putText(canvas, 'gt', (353, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25,255,1,cv2.LINE_AA)
            sample_lowout = lowout[64,:,:]
            image4 = np.transpose(sample_lowout*255)
            canvas[11:139, 428:556] = image4
            cv2.putText(canvas, 'lowout', (492, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25,255,1,cv2.LINE_AA)
            cv2.imwrite(result_path + str(num)+ '+' +filename + 'dim0'+'.png',np.uint8(canvas))



'公开数据集可视化'
def field_result_3d(args,net,num):         
    net.eval()
    with torch.no_grad():
        data_f3 = np.fromfile('/home/user/data/zwt/datasets/FaultDataset/f3d/gxl.dat', dtype=np.single)
        data_f3 = data_f3 - np.mean(data_f3)
        data_f3 = data_f3 / np.std(data_f3)
        input_data_f3 = np.reshape(data_f3,(512,384,128))

        data_kerry = np.load('/home/user/data/zwt/datasets/FaultDataset/f3d/Kerry_full.npy')
        data_kerry = data_kerry[:,:,0:280]
        data_kerry = data_kerry - np.mean(data_kerry)
        data_kerry = data_kerry / np.std(data_kerry)
        input_data_kerry = data_kerry

        data_shengli = np.load('/home/user/data/zwt/datasets/FaultDataset/ours/data_fi.npy')
        data_shengli = data_shengli[:,:,250:750]
        data_shengli = data_shengli - np.mean(data_shengli)
        data_shengli = data_shengli / np.std(data_shengli)
        input_data_shengli = data_shengli

        pred_path_f3 = './EXP/' + args.exp + '/results'+'/f3/' 
        if not os.path.exists(pred_path_f3):
            os.makedirs(pred_path_f3)

        pred_path_kerry = './EXP/' + args.exp + '/results'+'/kerry/' 
        if not os.path.exists(pred_path_kerry):
            os.makedirs(pred_path_kerry)

        pred_path_shengli = './EXP/' + args.exp + '/results'+'/shengli/' 
        if not os.path.exists(pred_path_shengli):
            os.makedirs(pred_path_shengli)

        # show_3d_result(input_data_f3, net, args.overlap, 128, 10, 10, 120, pred_path_f3,num)

        pos2_f3=125  
        pos1_f3=19 
        pos0_f3=15 
        pos2_kerry= 250
        pos1_kerry= 7
        pos0_kerry= 27
        pos2_shengli= 450
        pos1_shengli= 8
        pos0_shengli= 305

        block_size = (128, 128, 128)  # 切块大小
        output_data_f3 = sliding_window_prediction_3d(input_data_f3, net, block_size, args)
        show_result(np.transpose(input_data_f3[pos0_f3,:,:]),np.transpose(output_data_f3[pos0_f3,:,:]),pred_path_f3,num,0,pos0_f3,args)
        show_result(np.transpose(input_data_f3[:,pos1_f3,:]),np.transpose(output_data_f3[:,pos1_f3,:]),pred_path_f3,num,1,pos1_f3,args)
        show_result(np.transpose(input_data_f3[:,:,pos2_f3]),np.transpose(output_data_f3[:,:,pos2_f3]),pred_path_f3,num,2,pos2_f3,args)

        output_data_kerry = sliding_window_prediction_3d(input_data_kerry, net, block_size, args)
        show_result(np.transpose(input_data_kerry[pos0_kerry,:,:]),np.transpose(output_data_kerry[pos0_kerry,:,:]),pred_path_kerry,num,0,pos0_kerry,args)
        show_result(np.transpose(input_data_kerry[:,pos1_kerry,:]),np.transpose(output_data_kerry[:,pos1_kerry,:]),pred_path_kerry,num,1,pos1_kerry,args)
        show_result(np.transpose(input_data_kerry[:,:,pos2_kerry]),np.transpose(output_data_kerry[:,:,pos2_kerry]),pred_path_kerry,num,2,pos2_kerry,args)

        output_data_shengli = sliding_window_prediction_3d(input_data_shengli, net, block_size, args)
        show_result(np.transpose(input_data_shengli[pos0_shengli,:,:]),np.transpose(output_data_shengli[pos0_shengli,:,:]),pred_path_shengli,num,0,pos0_shengli,args)
        show_result(np.transpose(input_data_shengli[:,pos1_shengli,:]),np.transpose(output_data_shengli[:,pos1_shengli,:]),pred_path_shengli,num,1,pos1_shengli,args)
        show_result(np.transpose(input_data_shengli[:,:,pos2_shengli]),np.transpose(output_data_shengli[:,:,pos2_shengli]),pred_path_shengli,num,2,pos2_shengli,args)

