# Core Author: Zylo117
# Script's Author: winter2897

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import os
import math
import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
import os
from shutil import copyfile


def display(preds, imgs, ob_list):
    obj_list = ob_list
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        return imgs[i]


def excuteModel(videoname):
    # Video's path
    # set int to use webcam, set str to read from a video file

    if videoname is not None:
        video_src = os.path.join(r'D:\GitHub\Detection\server\uploads', f"{videoname}.mp4")
    else:
        video_src = 'D:\\GitHub\\Detection\\server\AImodel\\videotest\\default.mp4'

    compound_coef = 2
    trained_weights = 'D:\\GitHub\\Detection\\server\\AImodel\\weights\\efficientdet-video.pth'

    force_input_size = None  # set None to use default size

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # load model
    model = EfficientDetBackbone(
        compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(trained_weights))

    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # function for display

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        total = -1

    path_out = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'outvideo')

    path_result = r"D:\GitHub\Detection\server\AImodel\videotest\default.mp4"
    path_asset = r"D:\GitHub\Detection\client\src\assets"
    for i in range(0, length):
        ret, frame = cap.read()
        if not ret:
            break

        # frame preprocessing
        ori_imgs, framed_imgs, framed_metas = preprocess_video(
            frame, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda()
                             for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(
            0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

        # result
        out = invert_affine(framed_metas, out)
        img_show = display(out, ori_imgs, obj_list)

        if writer is None:

            # initialize our video writer
            fourcc = 0x00000021
            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if videoname is not None:
                path_result = os.path.join(path_out, f"{videoname}.mp4")
            else:
                path_result = os.path.join(path_out, "default.mp4")

            writer = cv2.VideoWriter(path_result, fourcc, 30, (img_show.shape[1], img_show.shape[0]), True)


        # write the output frame to disk
        writer.write(img_show)
        print("Processing data... " + str(round((i+1)/length, 3)*100) + " %")
        # show frame by frame
        #cv2.imshow('frame', img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] cleaning up...")

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if videoname is not None:
        path_asset = os.path.join(path_asset, f"{videoname}.mp4")
    else:
        path_asset = os.path.join(path_asset, "default.mp4")
    copyfile(path_result, path_asset)
    return path_asset
