import argparse
from sys import platform

from utils.datasets import *
from utils.utils import *
import torchvision


def detect(save_img=False):
    img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, view_img, save_txt = opt.output, opt.source, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    kwargs = {
        'box_score_thresh': opt.conf_thres,
    }
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=False)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=False)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        bboxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        # Process detections
        for i in range(len(labels)):
            p, im0 = path, im0s
            save_path = str(Path(out) / Path(p).name)

            if bboxes[i] is not None and len(bboxes[i]):
                # Rescale boxes from img_size to im0 size
                bboxes[i] = scale_coords_fasterrcnn(img.shape[2:], bboxes[i], im0.shape).round()

                # Write results
                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(labels[i])], scores[i])
                    plot_one_box(bboxes[i], im0, label=label, color=colors[int(labels[i])])

            # Print time (inference + NMS)
            print('Done. (%.3fs)' % (time.time() - t))

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', type=str, default='./coco_pt.names', help='*.names path')
    parser.add_argument('--source', type=str, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
