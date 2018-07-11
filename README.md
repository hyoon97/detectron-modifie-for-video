# detectron-modifie-for-video
This is a a project of a real-time Mask R-CNN using Detectron. The code given in this page is not receiving video input from webcam but from video file in use's computer. 

The modification done in detectron.py allows users to output a result in a video format and output coordinates of bbox in text files.

Exporting a result as a file allows users who installed detectron using docker and could not display the output. 

## Modifications

infer_simple.py (inside def main(args))
...
cap = cv2.VideoCapture('video.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    s = n(b'XVID')
    fourcc = cv2.VideoWriter_fourcc(*s)
    out = cv2.VideoWriter('output.avi', fourcc, 24.0, (width, height))
    im_name = 'tmp_im'
    count = 0
    fileOut = open('output.txt', 'w')

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
...

# Credit
https://github.com/facebookresearch/Detectron
https://github.com/cedrickchee/realtime-detectron
