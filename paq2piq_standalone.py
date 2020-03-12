import torch
import torch.nn as nn

import torchvision as tv
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.ops import RoIPool, RoIAlign

import numpy as np
from pathlib import Path
from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def get_idx(batch_size, n_output, device=None):
    idx = torch.arange(float(batch_size), dtype=torch.float, device=device).view(1, -1)
    idx = idx.repeat(n_output, 1, ).t()
    idx = idx.contiguous().view(-1, 1)
    return idx

def get_blockwise_rois(blk_size, img_size=None):
    if img_size is None: img_size = [1, 1]
    y = np.linspace(0, img_size[0], num=blk_size[0] + 1)
    x = np.linspace(0, img_size[1], num=blk_size[1] + 1)
    a = []
    for n in range(len(y) - 1):
        for m in range(len(x) - 1):
            a += [x[m], y[n], x[m + 1], y[n + 1]]
    return a

class RoIPoolModel(nn.Module):
    rois = None

    def __init__(self, backbone='resnet18'):
        super().__init__()
        if backbone is 'resnet18':
            model = tv.models.resnet18(pretrained=True)
            cut = -2
            spatial_scale = 1/32

        self.model_type = self.__class__.__name__
        self.body = nn.Sequential(*list(model.children())[:cut])
        self.head = nn.Sequential(
          AdaptiveConcatPool2d(),
          nn.Flatten(),
          nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.25, inplace=False),
          nn.Linear(in_features=1024, out_features=512, bias=True),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=512, out_features=1, bias=True)
        )
        self.roi_pool = RoIPool((2,2), spatial_scale)

    def forward(self, x):
        # compatitble with fastai model
        if isinstance(x, list) or isinstance(x, tuple):
            im_data, self.rois = x
        else:
            im_data = x

        feats = self.body(im_data)
        batch_size = im_data.size(0)

        if self.rois is not None:
            rois_data = self.rois.view(-1, 4)
            n_output = int(rois_data.size(0) / batch_size)
            idx = get_idx(batch_size, n_output, im_data.device)
            indexed_rois = torch.cat((idx, rois_data), 1)
            feats = self.roi_pool(feats, indexed_rois)
        preds = self.head(feats)
        return preds.view(batch_size, -1)

    def input_block_rois(self, blk_size=(20, 20), img_size=(1, 1), batch_size=1, include_image=True, device=None):
        a = [0, 0, img_size[1], img_size[0]] if include_image else []
        a += get_blockwise_rois(blk_size, img_size)
        t = torch.tensor(a).float().to(device)
        self.rois = t.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size, -1).view(-1, 4)


def normalize(x):
    x = np.array(x)
    x_mean, std_left, std_right = 72.59696108881171, 7.798274017370107, 4.118047289170692
    N_std = 3.5 # https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    # 3.5 -- > 99.9%
    # 3  -- > 99.7%
    x [x < x_mean] = x_mean + x_mean*(x[x < x_mean]-x_mean)/(N_std*std_left)
    x [x > x_mean] = x_mean + (100-x_mean)*(x[x > x_mean]-x_mean)/(N_std*std_right)
    # x [x < 0] = 0
    # x [x > 100] = 100
    return x.tolist()

class InferenceModel:
    blk_size = 20, 20
    categories = 'Bad', 'Poor', 'Fair', 'Good', 'Excellent'

    def __init__(self, model, path_to_model_state: Path):
        self.transform = transforms.ToTensor()
        model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
        self.model = model
        self.model.load_state_dict(model_state["model"])
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path: Path, render=False):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image: Image):
        image = image.convert("RGB")
        return self.predict(image)


    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        self.model.input_block_rois(self.blk_size, [image.shape[-2], image.shape[-1]], device=device)
        t = self.model(image).data.cpu().numpy()[0]

        local_scores = np.reshape(t[1:], self.blk_size)
        global_score = t[0]
        normed_score = normalize(global_score)

        return {"global_score": global_score,
                "normalized_global_score": normed_score,
                "local_scores": local_scores,
                "normalized_local_scores": normalize(local_scores),
                "category": self.categories[int(normed_score//20)]}
