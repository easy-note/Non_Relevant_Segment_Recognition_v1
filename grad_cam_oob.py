import os
import glob

import torch

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50

import torchvision.models as torch_models

from Model import CAMIO
from torchvision import transforms

from torchsummary import summary

from PIL import Image
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt

import argparse

import json

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
					choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d', 'mobilenet_v2', 'mobilenet_v3_small', 'squeezenet1_0'], help='trained backborn model, it will be yticks name')

parser.add_argument('--inference_img_dir', type=str, help='root dir for inference img')

parser.add_argument('--save_dir', type=str, help='gradcam results save path')

args, _ = parser.parse_known_args()


# for text on bar
def present_text(ax, bar, text, color='black'):
	for rect in bar:
		posx = rect.get_x()
		posy = rect.get_y()
		ax.text(posx, posy, text, size=17, color=color, rotation=0, ha='left', va='bottom')

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

    for img_idx, img_path in enumerate(all_inference_img_path) : 
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
        fig = plt.figure(figsize=(20,15))
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
        plt.savefig(os.path.join(save_dir, '{}_GRADCAM.jpg'.format(img_name)), format='jpg', dpi=200)
        plt.clf() # clear figure
        plt.cla() # clear axis

        # pil_image=Image.fromarray(visualization_IB)
        # pil_image.save(os.path.join(save_dir, 'grad_temp-{}.jpg'.format(idx)), format='JPEG')
    
    



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
    inference_img_dir = args.inference_img_dir
    save_dir = args.save_dir
    

    get_oob_grad_cam_img(model_path, model_name, inference_img_dir, save_dir)
    

