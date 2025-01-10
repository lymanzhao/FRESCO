import cv2
import torch.nn.functional as F
import numpy as np
from src.utils import *

def insert_key(keys, ind):
    """保持原始函数定义"""
    for i, k in enumerate(keys):
        if ind < k:
            keys.insert(i, ind)
            break
            
def get_maxinterv(keys):
    """保持原始函数定义"""
    maxinterv = 1
    for i in range(len(keys)-1):
        tmp = keys[i+1]-keys[i]
        if tmp > maxinterv:
            maxinterv = tmp
    return maxinterv

def get_motion_score(frame1, frame2):
    """内部辅助函数 - 计算运动分数"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude)

def get_keyframe_ind(filename, lastframen = 1e10, mininterv = 5, maxinterv = 20, viz = False):
    """保持原始函数签名和返回值类型"""
    if maxinterv == mininterv:
        return list(range(0,lastframen,mininterv))
    
    video_cap = cv2.VideoCapture(filename)
    n_frames = max(1, min(int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)), lastframen))
    
    # 使用与原代码相同的错误度量列表
    err = [0]
    preframe = None
    frames = []  # 存储处理后的帧
    
    # 保持原有的帧处理逻辑，但增加运动检测
    for i in range(n_frames):
        success, frame = video_cap.read()
        if not success:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, 512)
        img = cv2.GaussianBlur(img, (9, 9), 0.0)
        frames.append(img)
        
        if i == 0:
            preframe = numpy2tensor(img)
        else:
            curframe = numpy2tensor(img)
            # 组合MSE和运动分数
            mse_score = float(F.mse_loss(preframe, curframe).cpu().numpy())
            motion_score = get_motion_score(frames[-2], frames[-1])
            
            # 归一化并组合得分
            combined_score = mse_score + 0.5 * motion_score
            err += [combined_score]
            preframe = curframe
    
    video_cap.release()
    err = np.array(err)
    err1 = np.array(err)  # 保存原始错误用于可视化
    
    # 保持原有的关键帧选择逻辑框架
    n_frames = len(err)
    keys = [0, n_frames-1]
    err[0:mininterv] = -1
    err[-mininterv:] = -1
    
    # 使用改进的选择策略，但保持原有的循环结构
    while get_maxinterv(keys) > maxinterv:
        # 找到最显著的变化点
        ind = np.argmax(err)
        if err[ind] == -1:
            break
            
        # 检查局部最大值
        local_window = err[max(0, ind-mininterv):min(n_frames, ind+mininterv)]
        if err[ind] == max(local_window):
            # 在插入关键帧之前检查运动连续性
            if len(frames) > ind and ind > 0:
                motion_check = get_motion_score(frames[ind-1], frames[ind])
                if motion_check > np.mean(err1):  # 只在运动显著时插入
                    insert_key(keys, ind)
        
        # 维持最小间隔约束
        err[ind-mininterv:ind+mininterv] = -1
    
    # 可视化部分保持不变
    if viz:
        plt.plot(err1)
        plt.plot(keys, err1[keys], 'bo')
        plt.show()
    
    return keys
