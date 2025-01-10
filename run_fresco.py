import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "6"

import cv2
import io
import gc
import yaml
import argparse
import torch
import torchvision
import diffusers
from diffusers import FluxPipeline, AutoencoderKL, DDPMScheduler, ControlNetModel

from src.utils import *
from src.keyframe_selection import get_keyframe_ind
from src.diffusion_hacked import apply_FRESCO_attn, apply_FRESCO_opt, disable_FRESCO_opt
from src.diffusion_hacked import get_flow_and_interframe_paras, get_intraframe_paras
from src.pipe_FRESCO import inference

def get_models(config):
    print('\n' + '=' * 100)
    print('creating models...')
    import sys
    sys.path.append("./src/ebsynth/deps/gmflow/")
    sys.path.append("./src/EGNet/")
    sys.path.append("./src/ControlNet/")
    
    from gmflow.gmflow import GMFlow
    from model import build_model
    from annotator.canny import CannyDetector

    # optical flow - 保持不变
    flow_model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   ).to('cuda')
    
    checkpoint = torch.load(config['gmflow_path'], map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval() 
    print('create optical flow estimation model successfully!')
    
    # saliency detection - 保持不变
    sod_model = build_model('resnet')
    sod_model.load_state_dict(torch.load(config['sod_path']))
    sod_model.to("cuda").eval()
    print('create saliency detection model successfully!')
    
    # 修改为Canny ControlNet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                               torch_dtype=torch.float16)
    controlnet.to("cuda")
    detector = CannyDetector()
    print('create controlnet model-canny successfully!')
    
    # 替换为Flux模型
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(config['num_inference_steps'], device=pipe._execution_device)
    
    # 移除FreeU相关代码
    if config.get('use_freeu', False):
        print("FreeU is not supported in Flux models")
    
    # 保持FRESCO处理
    frescoProc = apply_FRESCO_attn(pipe)
    frescoProc.controller.disable_controller()
    apply_FRESCO_opt(pipe)
    print('create Flux model successfully!')
    
    # 冻结所有模型参数
    for param in flow_model.parameters():
        param.requires_grad = False    
    for param in sod_model.parameters():
        param.requires_grad = False
    for param in controlnet.parameters():
        param.requires_grad = False
    for param in pipe.unet.parameters():
        param.requires_grad = False
    
    return pipe, frescoProc, controlnet, detector, flow_model, sod_model

def apply_control(x, detector, config):
    # 简化为只使用Canny
    return detector(x, 50, 100)

def run_keyframe_translation(config):
    pipe, frescoProc, controlnet, detector, flow_model, sod_model = get_models(config)
    device = pipe._execution_device
    
    # Flux特定参数
    guidance_scale = 3.5  # Flux的默认值
    max_sequence_length = 512
    
    do_classifier_free_guidance = guidance_scale > 1
    assert(do_classifier_free_guidance)
    timesteps = pipe.scheduler.timesteps
    cond_scale = [config['cond_scale']] * config['num_inference_steps']
    dilate = Dilate(device=device)
    
    # 调整提示词处理
    base_prompt = config['prompt']
    a_prompt = ', best quality, extremely detailed, '
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing finger, extra digit, fewer digits, cropped, worst quality, low quality'    

    print('\n' + '=' * 100)
    print('key frame selection for \"%s\"...'%(config['file_path']))
    
    # 保持原有的关键帧选择逻辑
    video_cap = cv2.VideoCapture(config['file_path'])
    frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extra_prompts = [''] * frame_num
    keys = get_keyframe_ind(config['file_path'], frame_num, config['mininterv'], config['maxinterv'])
    
    # 创建输出目录
    os.makedirs(config['save_path'], exist_ok=True)
    os.makedirs(config['save_path']+'keys', exist_ok=True)
    os.makedirs(config['save_path']+'video', exist_ok=True)
    
    # 批处理逻辑保持不变
    sublists = [keys[i:i+config['batch_size']-2] for i in range(2, len(keys), config['batch_size']-2)]
    sublists[0].insert(0, keys[0])
    sublists[0].insert(1, keys[1])
    if len(sublists) > 1 and len(sublists[-1]) < 3:
        add_num = 3 - len(sublists[-1])
        sublists[-1] = sublists[-2][-add_num:] + sublists[-1]
        sublists[-2] = sublists[-2][:-add_num]

    if not sublists[-2]:
        del sublists[-2]
        
    print('processing %d batches:\nkeyframe indexes'%(len(sublists)), sublists)    

    print('\n' + '=' * 100)
    print('video to video translation...')
    
    batch_ind = 0
    propagation_mode = batch_ind > 0
    imgs = []
    record_latents = []
    video_cap = cv2.VideoCapture(config['file_path'])
    
    for i in range(frame_num):
        success, frame = video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, 512)
        H, W, C = img.shape
        Image.fromarray(img).save(os.path.join(config['save_path'], 'video/%04d.png'%(i)))
        if i not in sublists[batch_ind]:
            continue
            
        imgs += [img]
        if i != sublists[batch_ind][-1]:
            continue
        
        print('processing batch [%d/%d] with %d frames'%(batch_ind+1, len(sublists), len(sublists[batch_ind])))
        
        # 准备输入
        batch_size = len(imgs)
        n_prompts = [n_prompt] * len(imgs)
        prompts = [base_prompt + a_prompt + extra_prompts[ind] for ind in sublists[batch_ind]]
        if propagation_mode:
            assert len(imgs) == len(sublists[batch_ind]) + 2
            prompts = ref_prompt + prompts
        
        # 修改为Flux的prompt编码方式
        prompt_embeds = pipe._encode_prompt(
            prompts,
            device,
            max_sequence_length,
            do_classifier_free_guidance,
            n_prompts,
        )
            
        imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)
        edges = torch.cat([numpy2tensor(apply_control(img, detector, config)[:, :, None]) for img in imgs], dim=0)
        edges = edges.repeat(1,3,1,1).cuda() * 0.5 + 0.5
        if do_classifier_free_guidance:
            edges = torch.cat([edges.to(pipe.unet.dtype)] * 2)
            
        if config['use_salinecy']:
            saliency = get_saliency(imgs, sod_model, dilate) 
        else:
            saliency = None
        
        # 保持FRESCO的一致性控制
        flows, occs, attn_mask, interattn_paras = get_flow_and_interframe_paras(flow_model, imgs)
        correlation_matrix = get_intraframe_paras(pipe, imgs_torch, frescoProc, 
                            prompt_embeds, seed = config['seed'])
    
        frescoProc.controller.enable_controller(interattn_paras=interattn_paras, attn_mask=attn_mask)
        apply_FRESCO_opt(pipe, steps = timesteps[:config['end_opt_step']],
                         flows = flows, occs = occs, correlation_matrix=correlation_matrix, 
                         saliency=saliency, optimize_temporal = True)
        
        gc.collect()
        torch.cuda.empty_cache()   
        
        # 使用Flux进行推理
        latents = inference(pipe, controlnet, frescoProc, 
                  imgs_torch, prompt_embeds, edges, timesteps,
                  cond_scale, config['num_inference_steps'], config['num_warmup_steps'], 
                  do_classifier_free_guidance, config['seed'], guidance_scale, config['use_controlnet'],         
                  record_latents, propagation_mode,
                  flows = flows, occs = occs, saliency=saliency, 
                  repeat_noise=True,
                  max_sequence_length=max_sequence_length)  # 添加Flux特定参数

        gc.collect()
        torch.cuda.empty_cache()
        
        # 解码和保存结果
        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = torch.clamp(image, -1 , 1)
            save_imgs = tensor2numpy(image)
            bias = 2 if propagation_mode else 0
            for ind, num in enumerate(sublists[batch_ind]):
                Image.fromarray(save_imgs[ind+bias]).save(os.path.join(config['save_path'], 'keys/%04d.png'%(num)))
                
        gc.collect()
        torch.cuda.empty_cache()
        
        batch_ind += 1
        ref_prompt= [prompts[0], prompts[-1]]
        imgs = [imgs[0], imgs[-1]]
        propagation_mode = batch_ind > 0
        if batch_ind == len(sublists):
            gc.collect()
            torch.cuda.empty_cache()
            break    
    return keys

# run_full_video_translation函数保持不变
def run_full_video_translation(config, keys):
    # ... [保持原有代码不变]
    pass

if __name__ == '__main__':
    # main函数保持不变
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, 
                        default='./config/config_carturn.yaml',
                        help='The configuration file.')
    opt = parser.parse_args()

    print('=' * 100)
    print('loading configuration...')
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
        
    for name, value in sorted(config.items()):
        print('%s: %s' % (str(name), str(value)))  

    keys = run_keyframe_translation(config)
    run_full_video_translation(config, keys)
