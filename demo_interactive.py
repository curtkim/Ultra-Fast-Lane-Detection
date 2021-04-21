import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image

INPUT_DIR = '/data/rosbag/image2_resize/frame'
OUTPUT_DIR = 'tmp'

if __name__ == "__main__":
    
    img_w, img_h = 1280, 720
    row_anchor = tusimple_row_anchor

    args, cfg = merge_config()
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError


    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter('demo.avi', fourcc , 30.0, (img_w, img_h))
    
    for image_num in range(0, 483):
        name = INPUT_DIR + str(image_num).zfill(4) +'.jpg'
        print(name)
        im = Image.open(name)

        print("origin image", im.width, im.height)
        img = img_transforms(im)
        imgs = img.unsqueeze(0)   # [3, 288, 800] -> [1, 3, 288, 800]

        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)

        print('out.shape', out.shape) # [1, 101, 56, 4]

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc
      
        vis = cv2.imread(name)
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        #print(image_num, ppp)
                        cv2.circle(vis, ppp, 5, (0,255,0), -1)
        
        cv2.imwrite(OUTPUT_DIR + "/frame"+str(image_num).zfill(4)+".jpg", vis)

        vout.write(vis)

    vout.release()

    # 왜 VideoWriter가 동작 안하는지 모르겠다. 수동으로 ffmpeg를 돌림
    #ffmpeg -framerate 30 -i 'tmp/frame%04d.jpg' demo.mp4
