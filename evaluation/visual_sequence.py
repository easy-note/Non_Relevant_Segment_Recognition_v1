# IMG Sequence to GIF

from PIL import Image, ImageDraw



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
        images[0].save(results_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0) # 200 ms == 1/5 s, inf loop, no ommit img