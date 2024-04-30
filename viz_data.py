import argparse
import numpy as np 
import os
import glob 
import natsort 
from PIL import Image 
import shutil 
import cv2 
import tqdm 

class Obs:
    img = None 
    mask = None
    output = None

class Viz:
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Visualize Mask')
        parser.add_argument('--dir', type=str, default=None, required=True)
        parser.add_argument('--mode',type=str,choices=['cut','blue'],default='cut')
        parser.add_argument('--video',action='store_true')
        self.args =  parser.parse_args()

    def load_data(self):
        self.img_paths  = glob.glob(os.path.join(self.args.dir, 'Annotations','**','*.png'))
        self.img_paths = natsort.natsorted(self.img_paths)
        self.obs = []

        for img_path in self.img_paths:
            o = Obs()
            o.mask = np.array(Image.open(img_path))
            o.img = np.array(cv2.imread(img_path.replace('Annotations','JPEGImages')))
            self.obs.append(o)

    def process_all(self):
        for o in self.obs:
            o.output = self.process(o.img,o.mask)
    
    def process(self,img,mask):
        if self.args.mode == 'cut':
            return self.cut(img,mask)
        elif self.args.mode == 'blue':
            return self.blue(img,mask)

    def cut(self,img,mask):
        mask = mask[:,:,None].astype(np.float32)/255.0
        return img*mask

    def blue(self,img,mask):
        mask = mask[:,:].astype(np.float32)/255.0
        img = img.astype(np.float32)
        # make mask blue
        img[mask==1,2] = 255
        return img.astype(np.uint8)


    def save(self):
        output_dir = os.path.join(self.args.dir,self.args.mode)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for i,o in tqdm.tqdm(enumerate(self.obs)):
            cv2.imwrite(os.path.join(output_dir,f'{i}.png'),o.output)

        if self.args.video:
            os.system(f'ffmpeg -r 10 -i {output_dir}/%d.png -vcodec mpeg4 -y {output_dir}.mp4')


if __name__ == "__main__":
    v = Viz()
    v.parse_args()
    v.load_data()
    v.process_all()
    v.save()