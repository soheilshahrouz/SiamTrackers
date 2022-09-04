from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from nanotrack.core.config import cfg
from nanotrack.models.loss import select_cross_entropy_loss, select_iou_loss
from nanotrack.models.backbone import get_backbone
from nanotrack.models.head import get_ban_head
from nanotrack.models.neck import get_neck 

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.ban_head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        self.zf = zf

    def track(self, x):

        xf = self.backbone(x)

        cls, loc = self.ban_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
               } 

    def log_softmax(self, cls):

        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()

            cls = F.log_softmax(cls, dim=3)

        return cls 

    #  forward
    def forward(self, data):
        """ only used in training
        """
        # train mode
        if len(data)>=4: 
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            label_loc = data['label_loc'].cuda()

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)    

            cls, loc = self.ban_head(zf, xf)

            # cls loss with cross entropy loss 
            cls = self.log_softmax(cls)

            cls_loss = select_cross_entropy_loss(cls, label_cls) 

            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc, label_cls) 
            outputs = {} 

            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss 
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            return outputs  
        
        # test speed 
        else: 
        
            xf = self.backbone(data)  
            cls, loc = self.ban_head(self.zf, xf) 

            return {
                    'cls': cls,
                    'loc': loc,
                }



class NanoTrackTemplateMaker(nn.Module):
    def __init__(self, model):
        super(NanoTrackTemplateMaker, self).__init__()

        self.backbone = model.backbone

    def forward(self, z):
        z_perm = z.permute((0, 3, 1, 2))
        z_f = self.backbone(z_perm)
        return z_f



class BBPostProcessing(nn.Module):
    def __init__(self, size=16, stride=16):
        super(BBPostProcessing, self).__init__()

        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        
        self.points0 = nn.Parameter( torch.from_numpy(x.astype(np.float32).flatten()) )
        self.points1 = nn.Parameter( torch.from_numpy(y.astype(np.float32).flatten()) )

    def forward(self, delta):

        delta = delta.permute(1, 2, 3, 0).view(4, -1)

        delta0 = self.points0 - delta[0, :] #x1
        delta1 = self.points1 - delta[1, :] #y1
        delta2 = self.points0 + delta[2, :] #x2
        delta3 = self.points1 + delta[3, :] #y2

        x = (delta0 + delta2) * 0.5
        y = (delta1 + delta3) * 0.5
        w = delta2 - delta0
        h = delta3 - delta1

        return x, y, w, h


class NanoTrackForward(nn.Module):
    def __init__(self, model):
        super(NanoTrackForward, self).__init__()

        self.backbone = model.backbone
        self.ban_head = model.ban_head

        self.bb_pp = BBPostProcessing()


    def forward(self, x, z_f):
        
        x_perm = x.permute((0, 3, 1, 2))

        xf = self.backbone(x_perm)  
        cls, delta = self.ban_head(z_f, xf) 

        cls = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        cls = cls.softmax(1)[:, 1]

        x, y, w, h = self.bb_pp(delta)
        
        return x, y, w, h, cls