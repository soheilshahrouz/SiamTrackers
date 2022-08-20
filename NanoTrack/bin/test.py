# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

import sys 
sys.path.append(os.getcwd())  

from tqdm import tqdm 

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder, NanoTrackTemplateMaker, NanoTrackForward
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.bbox import get_axis_aligned_bbox
from nanotrack.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

# from bin.eval import eval

parser = argparse.ArgumentParser(description='nanotrack') 

parser.add_argument('--dataset', default='VOT2018', type=str,help='datasets')

parser.add_argument('--tracker_name', '-t', default='nanotrack',type=str,help='tracker name')

parser.add_argument('--config', default='./models/config/config.yaml',  type=str,help='config file')

parser.add_argument('--snapshot', default='./models/snapshot/checkpoint_e26.pth', type=str,help='snapshot of models to eval')

parser.add_argument('--save_path', default='./results', type=str, help='snapshot of models to eval')

parser.add_argument('--video', default='', type=str,  help='eval one special video')

parser.add_argument('--vis', action='store_true',help='whether v isualzie result')

parser.add_argument('--gpu_id', default='not_set', type=str, help="gpu id") 

parser.add_argument('--tracker_path', '-p', default='./results', type=str,help='tracker result path')

parser.add_argument('--num', '-n', default=4, type=int,help='number of thread to eval')

parser.add_argument('--show_video_level', '-s', dest='show_video_level',action='store_true')

parser.set_defaults(show_video_level=False)

args = parser.parse_args() 

if args.gpu_id != 'not_set': 

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

torch.set_num_threads(1)  

def main(): 
    
    cfg.merge_from_file(args.config) 

    dataset_root = os.path.join('./datasets', args.dataset) 
                  
    params = [0.0,0.0,0.0]
    
    params[0] =cfg.TRACK.LR 
    params[1]=cfg.TRACK.PENALTY_K
    params[2] =cfg.TRACK.WINDOW_INFLUENCE 

    params_name = args.snapshot.split('/')[-1] + ' '+ args.dataset + '  lr-' + str(params[0]) + '  pk-' + '_' + str(params[1]) + '  win-' + '_' + str(params[2])
    
    # create model 
    model = ModelBuilder() 

    # load model 
    model = load_pretrain(model, args.snapshot).cuda().eval()


    temp_model = NanoTrackTemplateMaker(model).cuda().eval()
    dummy_input = torch.randn(1, 127, 127, 3, device="cuda")
    input_names  = [ "template_maker_input" ]
    output_names = [ "template_maker_kernel_output"]

    torch.onnx.export(temp_model, dummy_input, "NanoTrack_Template_Maker.onnx", export_params=True,
        verbose=True, input_names=input_names, output_names=output_names)


    forw_model = NanoTrackForward(model).cuda()
    dummy_input = torch.randn(1, 255, 255, 3, device="cuda")
    dummy_kernel = torch.randn(1, 48, 8, 8, device="cuda")
    input_names  = [ "forward_input", "forward_kernel" ]
    output_names = [ "forward_delta0_output", "forward_delta1_output", "forward_delta2_output", "forward_delta3_output", "forward_cls_output" ]

    torch.onnx.export(forw_model, (dummy_input, dummy_kernel), "NanoTrack_Forward.onnx", export_params=True,
        verbose=True, input_names=input_names, output_names=output_names)
    
    exit()


    # build tracker 
    tracker = build_tracker(model)
    
    # create dataset 
    dataset = DatasetFactory.create_dataset(name=args.dataset,  
                                            dataset_root=dataset_root,
                                            load_img=False)  
    
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        total_lost=0
        avg_speed =0  
        for v_idx, video in tqdm(enumerate(dataset)):
            if args.video != '':
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0 
            pred_bboxes = [] 
            for idx, (img, gt_bbox) in enumerate(video): 
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                    gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter: 
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] #[topx,topy,w,h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        pred_bboxes.append(pred_bbox) 
                    else: 
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 
                        lost_number += 1 
                else:
                    pred_bboxes.append(0) 
                toc += cv2.getTickCount() - tic 
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                    (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency() 
            # save results
            video_path = os.path.join(args.save_path, args.dataset, args.tracker_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')

            total_lost += lost_number 
            avg_speed += idx / toc

        print('Speed: {:3.1f}fps'.format(avg_speed/60))
        print(params_name)
        
    else:
    # OPE tracking
        for v_idx, video in tqdm(enumerate(dataset)): 
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] #[topx,topy,w,h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset: 
                        pred_bboxes.append([1]) 
                    else: 
                        pred_bboxes.append(pred_bbox)
                else: 
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    #scores.append(outputs['best_score'])  
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0: 
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()

            # save results 
            if 'VOT2018-LT' == args.dataset: 
                video_path = os.path.join(args.save_path, args.dataset, args.tracker_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(args.save_path, args.dataset, args.tracker_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(args.save_path, args.dataset, args.tracker_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')   
    eval(args)   

if __name__ == '__main__':
    main()
