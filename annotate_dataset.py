import numpy as np 
import matplotlib.pyplot as plt
import cv2
from argparse import ArgumentParser
import glob 
import os
import torch 
import sys
from segment_anything import sam_model_registry, SamPredictor
import shutil
import json 
from PIL import Image
import copy 

class Viewpoint:
    def __init__(self):
        self.times = None 
        self.mask = None
        self.images = None 
        self.filenames = None 
        self.annotation = None 

class Tracker:
    def __init__(self):
        self.init_com = None 
        self.homography = None
        self.args=None
        self.SAM_points = []

    def parse_args(self):
        parser = ArgumentParser()
        parser.add_argument('--json', type=str, default='data')
        parser.add_argument('--device',choices=['cpu','cuda'], default='cuda')
        parser.add_argument('--annotate_scale',type=float,default=0.25)
        parser.add_argument('--output_dir', type=str, default='output')
        parser.add_argument('--checkpoint', type=str, default='sam_ckpts/sam_vit_l_0b3195.pth')
        parser.add_argument('--annotate',default=None,nargs='+',type=float)
        parser.add_argument('--n_points', type=int, default=5)

        self.args = parser.parse_args()
        self.device = torch.device(self.args.device)
        self.output_dir = self.args.output_dir
        self.input_dir = os.path.dirname(self.args.json)
        self.annotate_npy = os.path.join(self.output_dir,os.path.basename(self.args.json).replace('.json','_annotate.npy'))
        #if os.path.exists(self.output_dir):
            #shutil.rmtree(self.output_dir)
        #os.makedirs(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # delete all dirs in output dir
        for root, dirs, files in os.walk(self.output_dir):
            for d in dirs:
                shutil.rmtree(os.path.join(root,d))
        
        # annotations folder
        os.mkdir(os.path.join(self.output_dir,'Annotations'))
        # JPEGImages folder
        os.mkdir(os.path.join(self.output_dir,'JPEGImages'))

    def show_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image

    def load_SAM(self):
        sam_checkpoint = self.args.checkpoint # the checkpoint loaded in the setup section.
        model_type = "vit_h"
        if "vit_b" in sam_checkpoint:
            model_type = "vit_b"
        elif "vit_l" in sam_checkpoint:
            model_type = "vit_l"

        print(f"Loading model {model_type} from {sam_checkpoint}")
        device = "cuda" # loading to GPU.

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def load_data(self):
        if os.path.exists(self.annotate_npy):
            self.viewpoints = np.load(self.annotate_npy,allow_pickle=True)
        else: 
            # open json
            with open(self.args.json) as f:
                data = json.load(f)
                frames = data["frames"]
                self.transforms = np.unique([frame["transform_matrix"] for frame in frames],axis=0)
                self.times = np.unique([frame["time"] for frame in frames])
        
            self.viewpoints = []
            for i in range(len(self.transforms)):
                viewpoint = Viewpoint()
                viewpoint.times = []
                viewpoint.images = []
                viewpoint.filenames = []
                viewpoint.masks = []
                for frame in frames:
                    if np.all(frame["transform_matrix"] == self.transforms[i]):
                        viewpoint.times.append(frame["time"])
                        #viewpoint.images.append(cv2.imread(frame["filename"]))
                        viewpoint.filenames.append(frame["file_path"])
                        #viewpoint.masks.append(cv2.imread(frame["mask_filename"],0))
                # sort filenames by time
                viewpoint.filenames = [x for _,x in sorted(zip(viewpoint.times,viewpoint.filenames))]
                viewpoint.times = sorted(viewpoint.times)
                self.viewpoints.append(viewpoint)

    def get_SAM_annotations(self,image):

        print("Annoting point for initial image")
        self.ax.set_title('Select object, top left and bottom right corners.')
        if self.args.annotate_scale != 1.0:
            # resize image
            self.ax.imshow(cv2.resize(image, (0,0), fx=self.args.annotate_scale, fy=self.args.annotate_scale)/255.0)
        else:
            self.ax.imshow(image/255.0)
        self.SAM_points.append(np.array(plt.ginput(self.args.n_points)).flatten()/self.args.annotate_scale)
        print(self.SAM_points[-1])
        self.ax.cla()


    def viz_mask(self, mask, image,box):
        viz = self.show_mask(mask)[:,:,:3]
        image_viz = image.copy()/255.0
        image_viz[mask>0] = viz[mask>0]

        plt.imsave(os.path.join(self.output_dir,'viz.png'), image_viz) 
        # save mask as binary image 
        plt.imsave(os.path.join(self.output_dir,'mask.png'), mask, cmap='gray')


    def get_SAM_masks(self):
        for view in self.viewpoints:
            annnotations_dir = os.path.join(self.output_dir,'Annotations',os.path.dirname(view.filenames[0]))
            if not os.path.exists(annnotations_dir):
                os.makedirs(annnotations_dir)
            img_dir = os.path.join(self.output_dir,'JPEGImages',os.path.dirname(view.filenames[0]))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for i in range(len(view.filenames)):
                # copy file to output directory
                shutil.copy(os.path.join(self.input_dir,view.filenames[i]),os.path.join(img_dir,os.path.basename(view.filenames[i])))

            img = cv2.imread(os.path.join(self.input_dir,view.filenames[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(img)
            points = view.annotation.reshape((-1,2))
            mask = self.predictor.predict(
                point_coords=points,
                point_labels=np.ones(points.shape[0]),
                box=None,
                multimask_output=True,
                )[0][0] # hxw binary uint8 mask
            
            Image.fromarray(mask).save(os.path.join(annnotations_dir,os.path.basename(view.filenames[0])))

    def get_annotations(self):
        if self.args.annotate is not None:
            self.SAM_points = [np.array(self.args.annotate)]
            for i in range(len(self.viewpoints)):
                img = cv2.imread(os.path.join(self.input_dir,self.viewpoints[i].filenames[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                self.viewpoints[i].annotation = copy.deepcopy(self.SAM_points[-1])
                self.viewpoints[i].annotation[0] += w/2
                self.viewpoints[i].annotation[1] += h/2
        elif not os.path.exists(self.annotate_npy):
            # ask user to annotate four points in the image
            self.fig, self.ax = plt.subplots()
            for i in range(len(self.viewpoints)):
                img = cv2.imread(os.path.join(self.input_dir,self.viewpoints[i].filenames[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.get_SAM_annotations(img)
                self.viewpoints[i].annotation = self.SAM_points[-1]
        np.save(self.annotate_npy,self.viewpoints)

if __name__ == '__main__':
    tracker = Tracker()
    tracker.parse_args()
    tracker.load_SAM()
    tracker.load_data()
    tracker.get_annotations()
    tracker.get_SAM_masks()