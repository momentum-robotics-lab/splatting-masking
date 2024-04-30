import argparse
import numpy as np 
from PIL import Image

class Preprocessor:

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Preprocess Mask')
        parser.add_argument('--mask', type=str, default='mask.png', help='Path to mask image')
        parser.add_argument('--output', type=str, default='mask_new.png', help='Path to save mask')
        self.args =  parser.parse_args()


    def load_image(self):
        self.img =  np.array(Image.open(self.args.mask))
        if len(self.img.shape) > 2:
            self.img = self.img[:,:,0]
        breakpoint()

    def save_image(self):
        Image.fromarray(self.img).save(self.args.output)
if __name__ ==  '__main__':
    p = Preprocessor()
    p.parse_args()
    p.load_image()
    p.save_image()