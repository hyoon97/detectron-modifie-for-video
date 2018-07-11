#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform a simple inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.

Original source: https://github.com/facebookresearch/Detectron/blob/master/tools/infer_simple.py
"""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.utils import bytes_to_native_str as n

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import glob
import logging
import os
import sys
import time
# Set and get camera from OpenCV
#cam = cv2.VideoCapture('/detectron/mypython/people-walking.mp4')

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils



c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    """
    Add support for webcam
    """
    # Set and get camera from OpenCV
    cap = cv2.VideoCapture('/detectron/mypython/people-walking.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    s = n(b'XVID')
    fourcc = cv2.VideoWriter_fourcc(*s)
    out = cv2.VideoWriter('output.avi', fourcc, 24.0, (width, height))
    im_name = 'tmp_im'
    count = 0
    fileOut = open('people-walking.txt', 'w')

    while True:
        count += 1
        # Fetch image from camera
        _, im = cap.read()

        timers = defaultdict(Timer)
        t = time.time()

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        box_list = [b for b in cls_boxes if len(b) > 0]
        if len(box_list) > 0:
            boxes = np.concatenate(box_list)
        else:
            boxes = None
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            ext='jpg'  # default is PDF, but we want JPG.
        )
        time.sleep(0.05)
        img = cv2.imread('/detectron/mypython/tmp_im.jpg')
        cv2.putText(img, 'Frame: ',(5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(count),(130, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, 'Model: e2e_mask_rcnn_R-101-FPN_2x.yaml',(200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, 'WEIGHTS: https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-101.pkl',(5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for i in range(len(boxes)):
            x1 = "{:.6f}".format(boxes[i][0]/width)
            y1 = "{:.6f}".format(boxes[i][1]/height)
            x2 = "{:.6f}".format(boxes[i][2]/width)
            y2 = "{:.6f}".format(boxes[i][3]/height)
            conf = "{:.6f}".format(boxes[i][4])
            fileOut.write("Frame " + str(count).zfill(5) + ":" + "     " + str(x1) + "     " + str(y1) + "     " + str(x2) + "     " + str(y2) + "     " + str(conf) + "\n")
            #cv2.putText(img, str(x1),(5, 90+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            #cv2.putText(img, str(y1),(185, 90+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            #cv2.putText(img, str(x2),(365, 90+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            #cv2.putText(img, str(y2),(545, 90+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            #cv2.putText(img, str(conf),(725, 90+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        time.sleep(0.05)
        out.write(img)
    fileOut.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
