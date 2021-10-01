import os
from PIL import Image
import torch
import numpy as np
import argparse
from pandas import DataFrame as df
from tqdm import tqdm
import numpy as np

from decord import VideoReader
from decord import cpu, gpu
from PIL import Image

from torchvision import transforms
import pytorch_lightning as pl

from infer_train_model import CAMIO

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='inference model path (.ckpt)')
parser.add_argument('--input_path', type=str, help='input path (only support jpg, png, mp4, mpeg, avi). Recommand to use ch1 video.')  

args, _ = parser.parse_known_args()


# data transforms (output: tensor)
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
aug = data_transforms['test']
    
def batch_npy_to_tensor(batch_npy_img) : 
    """
    Convert batch numpy to 4D-tensor.
    Args:
        batch_npy_img: get_batch as numpy from video.
    Return:
        4D-tensor (batch, channel, height, width)
    """
    b, h, w, c = batch_npy_img.shape
    return_tensor = torch.Tensor(b, 3, 224, 224) # batch, channel, height, width
    
    # loop for Batch size
    for b_idx in range(b) : 
        # convert pil
        pil_img = Image.fromarray(batch_npy_img[b_idx])
        
        # pil to tensor ('using test' aug)
        return_tensor[b_idx] = aug(pil_img)
    
    return return_tensor

def npy_to_tensor(npy_img):
    """
    Convert numpy to 4D-tensor.
    Args:
        npy_img: image as numpy.
    Return:
        4D-tensor (batch, channel, height, width)
    """
    pil_img = Image.fromarray(npy_img)
    img = aug(pil_img)
    return_tensor = img.unsqueeze(0) # batch, channel, height, width
    
    return return_tensor

def trainer_load():
    """
    train model load and start inference.
    """
    # inference setting for each mode
    if 'mobilenet_v3_large' in args.model_path:
        backbone_model = 'mobilenet_v3_large'
    elif 'efficientnet_b3' in args.model_path:
        backbone_model = 'efficientnet_b3'

    test_hparams = {
        'optimizer_lr' : 0, # dummy (this option use only for training)
        'backbone_model' : backbone_model # args.model # (for train, test)
    }

    model = CAMIO.load_from_checkpoint(args.model_path, config=test_hparams)
    model.cuda()

    video_ext = ['mp4', 'mpeg', 'avi']
    frame_ext = ['jpg', 'png']

    if any(ext in args.input_path.lower() for ext in video_ext): # if video
        video = []
        video.append([args.input_path])
        inference_by_video(video, model)

    elif any(ext in args.input_path.lower() for ext in frame_ext): # if frame
        inference_by_frame(model)

    else:
        print('ERROR: not supported extention.')


def inference_by_video(video, model) :
    """
    inference using video. 
    Args:
        video: target video for infernece (args.input_path). we only support .mp4, .mpeg, .avi extention. -> str:
        model: trainer model. we only support .ckpt extention. -> str:
    Return:
        BATCH_PREDICT (predict list) -> List[int]:
    """
    # loop from total_videoset_cnt
    for i, input_path_list in enumerate(video, 1):
            
        # extract info for each input_path
        for input_path in input_path_list :
            
            video_name = os.path.splitext(os.path.basename(input_path))[0] # only video name

            video = VideoReader(input_path, ctx=cpu())
            video_len = len(video)

            # init_variable
            predict_list = [] # predict

            # for video inference, batch size
            BATCH_SIZE = 64
            
            # TOTAL_INFERENCE_FRAME_INDICES = list(range(0, video_len, inference_step))
            TOTAL_INFERENCE_FRAME_INDICES = list(range(0, video_len, 5))
            
            start_pos = 0
            end_pos = len(TOTAL_INFERENCE_FRAME_INDICES)
            
            with torch.no_grad() :
                model.eval()
                
                for idx in tqdm(range(start_pos, end_pos + BATCH_SIZE, BATCH_SIZE),  desc='Inferencing... \t ==> {}'.format(args.input_path)):
                    FRAME_INDICES = TOTAL_INFERENCE_FRAME_INDICES[start_pos:start_pos + BATCH_SIZE] # batch video frame idx

                    if FRAME_INDICES != []:
                        BATCH_INPUT = video.get_batch(FRAME_INDICES).asnumpy() # get batch (batch, height, width, channel)                        

                        # convert batch tensor
                        BATCH_TENSOR = batch_npy_to_tensor(BATCH_INPUT).cuda() # batch, channel, height, width

                        # inferencing model
                        BATCH_OUTPUT = model(BATCH_TENSOR)

                        # predict
                        BATCH_PREDICT = torch.argmax(BATCH_OUTPUT.cpu(), 1)
                        BATCH_PREDICT = BATCH_PREDICT.tolist()

                        # save results
                        predict_list+= list(BATCH_PREDICT)
                
                    start_pos = start_pos + BATCH_SIZE
                
                print(predict_list)
                return predict_list

def inference_by_frame(model):
    """
    inference using single frame. 
    Args:
        model: trainer model. we only support .ckpt extention. -> str:
    Return:
        BATCH_PREDICT (predict list) -> List[int]:
    """
    with torch.no_grad() :
        model.eval()

        FRAME_INDICES = [0]

        FRAME_INPUT = Image.open(args.input_path)
        FRAME_INPUT = np.array(FRAME_INPUT)

        # convert batch tensor
        BATCH_TENSOR = npy_to_tensor(FRAME_INPUT).cuda()

        # inferencing model
        BATCH_OUTPUT = model(BATCH_TENSOR)

        # predict
        BATCH_PREDICT = torch.argmax(BATCH_OUTPUT.cpu(), 1)
        BATCH_PREDICT = BATCH_PREDICT.tolist()

        print(BATCH_PREDICT)
        return BATCH_PREDICT


if __name__ == "__main__":
    ###  base setting for model testing ### 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    trainer_load()