import os
import glob

import torch

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50

import torchvision.models as torch_models

from train_model import CAMIO
from torchvision import transforms

from torchsummary import summary

from PIL import Image, ImageDraw
from torchvision import transforms

import numpy as np
import pandas as pd
from pandas import DataFrame as df

import matplotlib.pyplot as plt

import argparse

import json

import tqdm

from decord import VideoReader
from decord import cpu, gpu

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'gradcam' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]),
}

IB_CLASS, OOB_CLASS = [0,1]

parser = argparse.ArgumentParser()

parser.add_argument('--title_name', type=str, help='plot title, and save file name')

parser.add_argument('--model_path', type=str, help='model ckpt path')

parser.add_argument('--model_name', type=str,
                    choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d',
                    'mobilenet_v2', 'mobilenet_v3_small', 'squeezenet1_0', 'squeezenet1_1',
                    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'], help='backbone model')


parser.add_argument('--video_dir', type=str, help='Video Assets root dir')
parser.add_argument('--consensus_results_path', type=str, help='Inference-[ROBOT, LAPA]-[R_15, L_300] Assets dir')

# parser.add_argument('--inference_img_dir', type=str, help='root dir for inference img')

parser.add_argument('--save_dir', type=str, help='gradcam results save path')

args, _ = parser.parse_known_args()


# preprocessing pil image to modify resize or crop .. etc
def preprocessing_pil_img(im) :
    '''
    im:PIL
    '''
    
    processed_img = None

    ### 1. check img config info and set init var
    width, height = im.size

    ### 2. processing img    
    # 1. resize : 50 %
    processed_img = im.resize((int(width / 2), int(height / 2)))

    ### 3. return
    return processed_img

    

# make GIF file From IMG Sequence [https://note.nkmk.me/en/python-pillow-gif/]
def img_seq_to_gif(img_path_list, results_path) :
    '''
    img_path_list:[str, str ...]
    results_path:str {}.gif
    '''

    images = []

    ### 1. load img & seq append
    for img_path in img_path_list : 
        im = Image.open(img_path)

        # pre processing before add pil
        im = preprocessing_pil_img(im)
        
        images.append(im)

    ### 2. Saving Seq to GIF
    if len(images) <= 1 : # one img
        images[0].save(results_path)
    else :                # seq img
        images[0].save(results_path, save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0) # 200 ms == 1/2 s, inf loop, no ommit img


# input|DataFrame = 'frame' 'time' truth' 'predict'
# calc FN, FP, TP, TN
# out|{} = FN, FP, TP, TN frame
def return_metric_frame(result_df) :
    IB_CLASS, OOB_CLASS = 0,1

    print(result_df)
    
    # FN    
    FN_df = result_df[(result_df['truth']==OOB_CLASS) & (result_df['predict']==IB_CLASS)]
    
    # FP
    FP_df = result_df[(result_df['truth']==IB_CLASS) & (result_df['predict']==OOB_CLASS)]

    # TN
    TN_df = result_df[(result_df['truth']==IB_CLASS) & (result_df['predict']==IB_CLASS)]
    
    # TP
    TP_df = result_df[(result_df['truth']==OOB_CLASS) & (result_df['predict']==OOB_CLASS)]

    return {
        'FN_df' : FN_df,
        'FP_df' : FP_df,
        'TN_df' : TN_df,
        'TP_df' : TP_df,
    }

# group parsing
def return_group(metric_frame_df) :
    # return dict
    return_dict = {}

    # group by 'video_name'
    GROUP_BY_COL = 'video_name'
    group_names = list(metric_frame_df[GROUP_BY_COL].unique())
    
    # grouping
    for group in group_names :
        return_dict[group] = metric_frame_df.groupby(GROUP_BY_COL).get_group(group)
    
    return return_dict

# for text on bar
def present_text(ax, bar, text, color='black'):
	for rect in bar:
		posx = rect.get_x()
		posy = rect.get_y()
		ax.text(posx, posy, text, size=17, color=color, rotation=0, ha='left', va='bottom')


# save_dir = 'gradcam_results/mobilenet/robot_oob/R-17'
def get_oob_grad_cam_from_video(model_path, model_name, video_path, consensus_results_df, save_dir) : 

    ### 0. inference img to input tensor and log img to numpy
    print('TARGET VIDEO_PATH : ', video_path)
    video = VideoReader(video_path, ctx=cpu())

    total_df_len = len(consensus_results_df)
    
    input_tensor = torch.Tensor(total_df_len, 3, 224, 224) # float32 # for inference (softmax)
    rgb_img_list = [] # load rgb img # for gradcam
    
    
    # reset idx 하였으므로 df 순서대로 video_frame 획득
    print('RESET 전')
    print(consensus_results_df)
    consensus_results_df = consensus_results_df.reset_index()
    print('RESET 후')
    print(consensus_results_df)

    for df_idx in range(total_df_len) :
        frame_idx = consensus_results_df.iloc[df_idx, :]['frame'] # parsing info 'frame'
        print(df_idx, '|\t', frame_idx)
        
        numpy_img = video[frame_idx].asnumpy() # numpy # uint8
        pil_img = Image.fromarray(numpy_img) # convert pil

        rgb_img = np.array(data_transforms['gradcam'](pil_img)).astype(np.float32) / 255.0 # normalize pixel intensity [0-1]
        
        input_tensor[df_idx] = data_transforms['test'](pil_img) # input tensor
        rgb_img_list.append(rgb_img)

    del video
    
    ### 1. load model
    test_hparams = {
        'optimizer_lr' : 0, # dummy (this option use only for training)
        'backborn_model' : model_name # (for train, test)
    }

    models = CAMIO.load_from_checkpoint(model_path, config=test_hparams)

    models.cuda()
    models.eval()

    ### 2. summary model
    print('\n\n==== MODEL CLASS ====\n\n')
    print(models)
    print('\n\n==== MODEL SUMMARY ====\n\n')
    summary(models.model, (3,224,224))

    # Model predict
    with torch.no_grad() :
        outputs = models(input_tensor.cuda())


    predict = torch.nn.Softmax(dim=1)(outputs.cpu()) # softmax
    print(torch.argmax(predict.cpu(), 1)) # predict class
    

    ### 3. select gradcam layer
    if (model_name.find('resnet') != -1) or (model_name.find('resnext') != -1) :
        if model_name == 'resnet18' :
            print('MODEL = RESNET18')
            target_layer = None

        elif model_name == 'resnet34' :
            print('MODEL = RESNET34')
            target_layer = None
            
        elif model_name == 'resnet50' :
            print('MODEL = RESNET50')
            target_layer = None
            
        elif model_name == 'wide_resnet50_2':
            print('MODEL = WIDE_RESNET50_2')
            target_layer = models.model.layer4[-1] # layer4 // bottleneck(2) // conv3

        elif model_name == 'resnext50_32x4d':
            print('MODEL = RESNEXT50_32x4D')
            target_layer = models.model.layer4[-1] # layer4 // bottleneck(2) // conv3
            
        else : 
            assert(False, '=== Not supported Resnet model ===')

    elif model_name.find('mobilenet') != -1 :
        if model_name == 'mobilenet_v2' :
            print('MODEL = MOBILENET_V2')
            target_layer = None

        elif model_name == 'mobilenet_v3_small' :
            print('MODEL = MOBILENET_V3_SMALL')
            target_layer = models.model.features[-1][-1] # ConvBNActivation // Hardwish
            
        else :
            assert(False, '=== Not supported MobileNet model ===')
    
    elif model_name.find('squeezenet') != -1 :
        if model_name == 'squeezenet1_0' :
            print('MODEL = squeezenet1_0')
            target_layer = models.model.classifier[0]

        else :
            assert(False, '=== Not supported Squeezenet model ===')

    elif model_name.find('efficientnet') != -1 :
        if model_name == 'efficientnet_b0' :
            print('MODEL = EFFICIENTNET-B0')
            target_layer = None

        elif model_name == 'efficientnet_b1' :
            print('MODEL = EFFICIENTNET-B1')
            target_layer = None

        elif model_name == 'efficientnet_b2' :
            print('MODEL = EFFICIENTNET-B2')
            target_layer = None

        elif model_name == 'efficientnet_b3' :
            print('MODEL = EFFICIENTNET-B3')
            target_layer = models.model._blocks[-1] # final MBConvBlock
            # target_layer = models.model._conv_head

        elif model_name == 'efficientnet_b4' :
            print('MODEL = EFFICIENTNET-B4')
            target_layer = None

        else :
            assert(False, '=== Not supported Efficient model ===')

    
    else :
        assert(False, '=== Not supported Model === ')

    ### 3. select gradcam layer
    # target_layer = models.model.classifier[0] # final conv (squeezenet)
    print('\n\n==== TARGET LAYER ====\n\n')
    print(target_layer)


    ### 4. GradCAM
    cam = GradCAM(model=models.model, target_layer=target_layer, use_cuda=True)

    # target_category = OOB_CLASS
    #### 4_1. CLASS_IDX 별 Gradcam

    # init batch size and Gradcam output
    BATCH_SIZE = 32
    grayscale_cam_IB = np.empty((1,224,224)) # (batch, w, h) or (batch, h, w)
    grayscale_cam_OOB = np.empty((1,224,224))

    for i in range((len(input_tensor) + BATCH_SIZE - 1) // BATCH_SIZE ) :
        batch_input_tensor = input_tensor[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        
        batch_grayscale_cam_IB = cam(input_tensor=batch_input_tensor.cuda(), target_category=IB_CLASS)
        batch_grayscale_cam_OOB = cam(input_tensor=batch_input_tensor.cuda(), target_category=OOB_CLASS)

        grayscale_cam_IB = np.append(grayscale_cam_IB, batch_grayscale_cam_IB, axis=0)
        grayscale_cam_OOB = np.append(grayscale_cam_OOB, batch_grayscale_cam_OOB, axis=0)

        print(batch_grayscale_cam_IB)
        print(type(batch_grayscale_cam_IB))
        print(batch_grayscale_cam_IB.shape)
        print(batch_grayscale_cam_IB.dtype)
        


    ### 5. gradscale cam
    
    print(grayscale_cam_IB)
    print(type(grayscale_cam_IB))
    print(grayscale_cam_IB.shape)
    print(grayscale_cam_IB.dtype)
    
    

    # grayscale_cam = grayscale_cam[0, :] # 0번째 batch
    
    ### 6. visualization
    for df_idx in range(total_df_len) : # rgb_img와 tesnor가 해당순서대로 append 되어 있으므로 df_idx 순서
        # 해당 frame 정보
        df_row = consensus_results_df.iloc[df_idx, :] # df_idx row info

        img_name = '{}-{:010d}'.format(df_row['video_name'], df_row['frame'])
        
        print('{}\t/{}\t{}'.format(df_idx, total_df_len, img_name))
        # print(grayscale_cam_IB[df_idx, :].shape)
        # print(rgb_img_list[df_idx].shape)

        # https://github.com/jacobgil/pytorch-grad-cam/blob/137dbd18df363ac0fc8af9df9091f098aaf3c2b6/pytorch_grad_cam/base_cam.py#L27
        visualization_IB = show_cam_on_image(rgb_img_list[df_idx], grayscale_cam_IB[df_idx, :], use_rgb=True) # numpy[0-1], numpy
        visualization_OOB = show_cam_on_image(rgb_img_list[df_idx], grayscale_cam_OOB[df_idx, :], use_rgb=True) # numpy[0-1], numpy

        # print(visualization_IB.shape)
        # print(visualization_IB.dtype)

        #######  plt setting #######
        fig = plt.figure(figsize=(18,12))
        label_names = ('IB', 'OOB')
        colors = ('cadetblue', 'orange')
        
        #### 1. fig title
        fig.suptitle('{}'.format(args.title_name), fontsize=20)

        #### 2. shape, location, rowspan, colspane
        ax1 = plt.subplot2grid((3,6), (0,0), rowspan=2, colspan=2) # Input 
        ax2 = plt.subplot2grid((3,6), (0,2), rowspan=2, colspan=2) # IB
        ax3 = plt.subplot2grid((3,6), (0,4), rowspan=2, colspan=2) # OOB
        ax4 = plt.subplot2grid((3,6), (2,0), colspan=5) # Softmax        

        #### 3. imshow
        ax1.imshow(rgb_img_list[df_idx])
        ax2.imshow(visualization_IB)
        ax3.imshow(visualization_OOB)

        #### 4. barh | softmax
        bar = ax4.barh(IB_CLASS, predict[df_idx][IB_CLASS], label='IB', color=colors[IB_CLASS], height=0.5)
        present_text(ax4, bar, '{:.3f}'.format(float(predict[df_idx][IB_CLASS])))
        bar = ax4.barh(OOB_CLASS, predict[df_idx][OOB_CLASS], label='OOB', color=colors[OOB_CLASS], height=0.5)
        present_text(ax4, bar, '{:.3f}'.format(float(predict[df_idx][OOB_CLASS])))

        #### 5. x축 세부설정
        ax4.set_xticks(np.arange(0, 1, step=0.1))
        ax4.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0, 1, step=0.1)])
        ax4.xaxis.set_tick_params(labelsize=10)
        ax4.set_xlabel('SOFTMAX OUTPUT', fontsize=12)

        #### 6. y축 세부설정
        ax4.set_yticks((IB_CLASS, OOB_CLASS))
        ax4.set_yticklabels(label_names, fontsize=10)	
        ax4.set_ylabel('CLASS', fontsize=12)

        #### 7. 보조선(눈금선) 나타내기
        ax4.set_axisbelow(True)
        ax4.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

        #### 8. 범례 나타내기
        box = ax4.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax4.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax4.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # print('{}\nFRAME : {} \tCONSENEUS_FRAME : {}\nTIME : {}\tCONSENSUS_TIME : {}\nTRUTH : {}\tPREDICT : {}'.format(df_row['video_name'], df_row['frame'], df_row['consensus_frame'], df_row['time'], df_row['consensus_time'], df_row['truth'], df_row['predict']))

        #### 9. 개별 title 설정
        ax1.set_title('Input',  fontsize=15)
        ax2.set_title('IB',  fontsize=15)
        ax3.set_title('OOB',  fontsize=15)
        ax4.set_title('{}\nFRAME : {} - CONSENEUS_FRAME : {}\nTIME : {} - CONSENSUS_TIME : {}\nTRUTH : {} - PREDICT : {}'.format(df_row['video_name'], df_row['frame'], df_row['consensus_frame'], df_row['time'], df_row['consensus_time'], df_row['truth'], df_row['predict']),  fontsize=18)

        ## 10. fig tight 설정
        fig.tight_layout() # subbplot 간격 줄이기

        ### 11. save img
        plt.show()
        plt.savefig(os.path.join(save_dir, '{}-GRADCAM.jpg'.format(img_name)), format='jpg', dpi=100)
        plt.clf() # clear figure
        plt.cla() # clear axis

        # pil_image=Image.fromarray(visualization_IB)
        # pil_image.save(os.path.join(save_dir, 'grad_temp-{}.jpg'.format(idx)), format='JPEG')


def get_oob_grad_cam_img(model_path, model_name, inference_img_dir, save_dir) : 

    ### 0. inference img to input tensor and log img to numpy
    all_inference_img_path = sorted(glob.glob(inference_img_dir +'/*{}'.format('jpg'))) # all inference file list
    input_tensor = torch.Tensor(len(all_inference_img_path), 3, 224, 224) # float32
    rgb_img_list = [] # load rgb img


    for idx, img_path in enumerate(all_inference_img_path) : 
        img = Image.open(img_path)
        rgb_img = np.array(data_transforms['gradcam'](img)).astype(np.float32)
        rgb_img /= 255.0 # normalize pixel intensity [0-1]

        input_tensor[idx] = data_transforms['test'](img) # input tensor
        rgb_img_list.append(rgb_img) # rgb img

    print(input_tensor)
    print('size : ', input_tensor.shape)
    print('type : ', input_tensor.dtype)

    
    ### 1. load model
    test_hparams = {
        'optimizer_lr' : 0, # dummy (this option use only for training)
        'backborn_model' : model_name # (for train, test)
    }

    models = CAMIO.load_from_checkpoint(model_path, config=test_hparams)

    models.cuda()
    models.eval()

    ### 2. summary model
    print('\n\n==== MODEL CLASS ====\n\n')
    print(models)
    print('\n\n==== MODEL SUMMARY ====\n\n')
    summary(models.model, (3,224,224))

    # Model predict
    with torch.no_grad() :
        outputs = models(input_tensor.cuda())


    predict = torch.nn.Softmax(dim=1)(outputs.cpu()) # softmax
    print(torch.argmax(predict.cpu(), 1)) # predict class
    

    ### 3. select gradcam layer
    if (model_name.find('resnet') != -1) or (model_name.find('resnext') != -1) :
        if model_name == 'resnet18' :
            print('MODEL = RESNET18')
            target_layer = None

        elif model_name == 'resnet34' :
            print('MODEL = RESNET34')
            target_layer = None
            
        elif model_name == 'resnet50' :
            print('MODEL = RESNET50')
            target_layer = None
            
        elif model_name == 'wide_resnet50_2':
            print('MODEL = WIDE_RESNET50_2')
            target_layer = models.model.layer4[-1] # layer4 // bottleneck(2) // conv3

        elif model_name == 'resnext50_32x4d':
            print('MODEL = RESNEXT50_32x4D')
            target_layer = models.model.layer4[-1] # layer4 // bottleneck(2) // conv3
            
        else : 
            assert(False, '=== Not supported Resnet model ===')

    elif model_name.find('mobilenet') != -1 :
        if model_name == 'mobilenet_v2' :
            print('MODEL = MOBILENET_V2')
            target_layer = None

        elif model_name == 'mobilenet_v3_small' :
            print('MODEL = MOBILENET_V3_SMALL')
            target_layer = models.model.features[-1][-1] # ConvBNActivation // Hardwish
            
        else :
            assert(False, '=== Not supported MobileNet model ===')
    
    elif model_name.find('squeezenet') != -1 :
        if model_name == 'squeezenet1_0' :
            print('MODEL = squeezenet1_0')
            target_layer = models.model.classifier[0]

        else :
            assert(False, '=== Not supported Squeezenet model ===')

    
    else :
        assert(False, '=== Not supported Model === ')

    ### 3. select gradcam layer
    # target_layer = models.model.classifier[0] # final conv (squeezenet)
    print('\n\n==== TARGET LAYER ====\n\n')
    print(target_layer)


    ### 4. GradCAM
    cam = GradCAM(model=models.model, target_layer=target_layer, use_cuda=True)

    # target_category = OOB_CLASS
    #### 4_1. CLASS_IDX 별 Gradcam

    # init batch size and Gradcam output
    BATCH_SIZE = 32
    grayscale_cam_IB = np.empty((1,224,224)) # (batch, w, h) or (batch, h, w)
    grayscale_cam_OOB = np.empty((1,224,224))

    for i in range((len(input_tensor) + BATCH_SIZE - 1) // BATCH_SIZE ) :
        batch_input_tensor = input_tensor[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        
        batch_grayscale_cam_IB = cam(input_tensor=batch_input_tensor.cuda(), target_category=IB_CLASS)
        batch_grayscale_cam_OOB = cam(input_tensor=batch_input_tensor.cuda(), target_category=OOB_CLASS)

        grayscale_cam_IB = np.append(grayscale_cam_IB, batch_grayscale_cam_IB, axis=0)
        grayscale_cam_OOB = np.append(grayscale_cam_OOB, batch_grayscale_cam_OOB, axis=0)

        print(batch_grayscale_cam_IB)
        print(type(batch_grayscale_cam_IB))
        print(batch_grayscale_cam_IB.shape)
        print(batch_grayscale_cam_IB.dtype)
        


    ### 5. gradscale cam
    print(grayscale_cam_IB)
    print(type(grayscale_cam_IB))
    print(grayscale_cam_IB.shape)
    print(grayscale_cam_IB.dtype)
    

    # grayscale_cam = grayscale_cam[0, :] # 0번째 batch
    
    ### 6. visualization

    for img_idx, img_path in tqdm(enumerate(all_inference_img_path)) : 
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        print(img_name)
        print(grayscale_cam_IB[img_idx, :].shape)
        print(rgb_img_list[img_idx].shape)

        # https://github.com/jacobgil/pytorch-grad-cam/blob/137dbd18df363ac0fc8af9df9091f098aaf3c2b6/pytorch_grad_cam/base_cam.py#L27
        visualization_IB = show_cam_on_image(rgb_img_list[img_idx], grayscale_cam_IB[img_idx, :], use_rgb=True) # numpy[0-1], numpy
        visualization_OOB = show_cam_on_image(rgb_img_list[img_idx], grayscale_cam_OOB[img_idx, :], use_rgb=True) # numpy[0-1], numpy

        print(visualization_IB.shape)
        print(visualization_IB.dtype)

        #######  plt setting #######
        fig = plt.figure(figsize=(18,10))
        label_names = ('IB', 'OOB')
        colors = ('cadetblue', 'orange')
        
        #### 1. fig title
        fig.suptitle('{}'.format(args.title_name), fontsize=25)

        #### 2. shape, location, rowspan, colspane
        ax1 = plt.subplot2grid((3,6), (0,0), rowspan=2, colspan=2) # Input 
        ax2 = plt.subplot2grid((3,6), (0,2), rowspan=2, colspan=2) # IB
        ax3 = plt.subplot2grid((3,6), (0,4), rowspan=2, colspan=2) # OOB
        ax4 = plt.subplot2grid((3,6), (2,0), colspan=5) # Softmax        

        #### 3. imshow
        ax1.imshow(rgb_img_list[img_idx])
        ax2.imshow(visualization_IB)
        ax3.imshow(visualization_OOB)

        #### 4. barh | softmax
        bar = ax4.barh(IB_CLASS, predict[img_idx][IB_CLASS], label='IB', color=colors[IB_CLASS], height=0.5)
        present_text(ax4, bar, '{:.3f}'.format(float(predict[img_idx][IB_CLASS])))
        bar = ax4.barh(OOB_CLASS, predict[img_idx][OOB_CLASS], label='OOB', color=colors[OOB_CLASS], height=0.5)
        present_text(ax4, bar, '{:.3f}'.format(float(predict[img_idx][OOB_CLASS])))

        #### 5. x축 세부설정
        ax4.set_xticks(np.arange(0, 1, step=0.1))
        ax4.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(0, 1, step=0.1)])
        ax4.xaxis.set_tick_params(labelsize=10)
        ax4.set_xlabel('SOFTMAX OUTPUT', fontsize=12)

        #### 6. y축 세부설정
        ax4.set_yticks((IB_CLASS, OOB_CLASS))
        ax4.set_yticklabels(label_names, fontsize=10)	
        ax4.set_ylabel('CLASS', fontsize=12)

        #### 7. 보조선(눈금선) 나타내기
        ax4.set_axisbelow(True)
        ax4.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

        #### 8. 범례 나타내기
        box = ax4.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax4.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax4.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        #### 9. 개별 title 설정
        ax1.set_title('Input',  fontsize=15)
        ax2.set_title('IB',  fontsize=15)
        ax3.set_title('OOB',  fontsize=15)
        ax4.set_title('{}'.format(img_name),  fontsize=18)

        ## 10. fig tight 설정
        fig.tight_layout() # subbplot 간격 줄이기

        ### 11. save img
        plt.show()
        plt.savefig(os.path.join(save_dir, '{}_GRADCAM.jpg'.format(img_name)), format='jpg', dpi=100)
        plt.clf() # clear figure
        plt.cla() # clear axis

        # pil_image=Image.fromarray(visualization_IB)
        # pil_image.save(os.path.join(save_dir, 'grad_temp-{}.jpg'.format(idx)), format='JPEG')
    
    

def main(model_path, model_name, video_dir, consensus_results_path, save_dir) :
    
    # video_dir의 모든 video parsing
    all_video_path = []
    video_ext_list = ['mp4', 'MP4', 'mpg']

    for ext in video_ext_list :
        all_video_path.extend(glob.glob(video_dir +'/*.{}'.format(ext)))

    # consensus_results csv loading
    consensus_results_df = pd.read_csv(consensus_results_path)
    
    # calc FP frame
    metric_frame_dict = return_metric_frame(consensus_results_df)
    FP_consensus_df = metric_frame_dict['FP_df']

    # group by 'video_name'
    FP_df_per_group = return_group(FP_consensus_df)
    
    # video_name별로 FP Frame Gradcam
    for video_name, FP_df in FP_df_per_group.items() :
        print('video_name : ', video_name)

        # video_dir 찾기
        video_path_list = [v_path for v_path in all_video_path if video_name in v_path]

        # video_dir 찾은 Video가 1개일 경우에만 처리
        if len(video_path_list) == 1 :
            video_path = video_path_list[0]

            # set save dir and make dir
            each_save_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0])
            try :
                if not os.path.exists(each_save_dir) :
                    os.makedirs(each_save_dir)
            except OSError :
                print('ERROR : Creating Directory, ' + each_save_dir)

            # gradcam visual save in each save_dir
            get_oob_grad_cam_from_video(model_path, model_name, video_path, FP_df, each_save_dir)

            # save_dir img to gif
            print('\n\n===> CONVERTING GIF\n\n')
            all_results_img_path = sorted(glob.glob(each_save_dir +'/*{}'.format('jpg'))) # 위에서 저장한 img 모두 parsing
            img_seq_to_gif(all_results_img_path, os.path.join(each_save_dir, '{}-GRADCAM.gif'.format(video_name))) # seqence 이므로 sort 하여 append
            print('\n\n===> DONE\n\n')
    
        else : # 비디오 여러개일 경우 오류
            assert False, "ERROR : Duplicatied Video Exist"



if __name__ == '__main__' :

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    '''
    model_path = './logs/ROBOT/OOB/robot-oob-0423-fold_2/ckpoint_robot-oob-0423-fold_2-model=wide_resnet50_2-batch=32-lr=0.001-fold=2-ratio=3-epoch=49-last.ckpt'
    model_name = 'wide_resnet50_2'
    inference_img_dir = './results-robot_oob-wide_resnet50_2-fold_2-last/R006/R006_ch1_video_01/fp_frame'
    save_dir = './gradcam_results'
    '''

    # print args
    print(json.dumps(args.__dict__, indent=2))

    model_path = args.model_path
    model_name = args.model_name
    # inference_img_dir = args.inference_img_dir
    video_dir = args.video_dir
    consensus_results_path = args.consensus_results_path
    save_dir = args.save_dir


    main(model_path, model_name, video_dir, consensus_results_path, save_dir)
    # get_oob_grad_cam_img(model_path, model_name, inference_img_dir, save_dir)
