# SSIM : Structural Similarity Index Measure

def using_skimage(path1, path2):
    '''
    출처 : https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
    '''
    from skimage.metrics import structural_similarity as ssim
    import argparse
    import imutils
    import cv2

    # Load the two input images
    imageA = cv2.imread(path1)
    imageB = cv2.imread(path2)

    #    Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    #    Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # You can print only the score if you want
    print("[gray] using Skimage: {}".format(score))
    print('\n')

    ########### [B, G, R] 따로 계산 ###########
    #    Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    
    # b_a = imageA[:,:,0]
    # g_a = imageA[:,:,1]
    # r_a = imageA[:,:,2]

    # b_b = imageB[:,:,0]
    # g_b = imageB[:,:,1]
    # r_b = imageB[:,:,2]

    b_a, g_a, r_a = cv2.split(imageA)
    b_b, g_b, r_b = cv2.split(imageB)

    (score_b, diff_b) = ssim(b_a, b_b, full=True)
    diff_b = (diff_b * 255).astype("uint8")

    (score_g, diff_g) = ssim(g_a, g_b, full=True)
    diff_g = (diff_g * 255).astype("uint8")

    (score_r, diff_r) = ssim(r_a, r_b, full=True)
    diff_r = (diff_r * 255).astype("uint8")

    # You can print only the score if you want
    print("using Skimage (b, g, r): {}, {}, {}".format(score_b, score_g, score_r))
    print("using Skimage avg: {}".format((score_b+score_g+score_r)/3))
    print("using Skimage avg: {}".format((score_b*0.114+score_g*0.589+score_r*0.299)))
    print('\n')


def using_iqa_pytorch(path1, path2):
    '''
    출처 : https://bskyvision.com/878
    '''
    from IQA_pytorch import SSIM, utils
    from PIL import Image
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    ref = utils.prepare_image(Image.open(path1).convert("RGB")).to(device)
    dist = utils.prepare_image(Image.open(path2).convert("RGB")).to(device)
    
    model = SSIM(channels=3)
    
    score = model(dist, ref, as_loss=False)
    print('using IQA_pytorch : %.4f' % score.item())
    print('\n')



def using_ssim_pil(path1, path2):
    '''
    출처 : https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
    '''
    from SSIM_PIL import compare_ssim
    from PIL import Image

    image1 = Image.open(path1)
    image2 = Image.open(path2)

    value = compare_ssim(image1, image2) # Compare images using OpenCL by default
    print('using SSIM_Pil : ', value)



if __name__ == '__main__':
    path1 = '/OOB_RECOG/template/core/dataset/SSIM/img/04_GS4_99_L_1_01-0000000000.jpg'
    path2 = '/OOB_RECOG/template/core/dataset/SSIM/img/04_GS4_99_L_1_01-0000000200.jpg'
    
    using_cv2(path1, path2)
    using_iqa_pytorch(path1, path2)
    ssim_ssim_pil(path1, path2)





