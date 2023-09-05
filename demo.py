"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse
import numpy as np

import matplotlib.pyplot as plt

from torch.nn import DataParallel as DP
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, checkpoint,  optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # save video with 12 fps
    fps = 12

    # load network
    net_w = net_h = 384
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        attn_interval=3,
    )

    model = DP(model)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model'])

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    video_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(video_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, video_name in enumerate(video_names):

        if os.path.isdir(video_name):
            continue

        print("  processing {} ({}/{})".format(video_name, ind + 1, num_images))
        # input

        filename = os.path.join(
            output_path, os.path.basename(video_name)
        )
        vidcap = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # img_inputs = []
        img_inputs = None
        predictions = None

        overlap = 1
        interval = 1
        seq_len = 4
        shift = 0

        valid_frame = 0
        while 1:
            success, image = vidcap.read()
            if success is False:
                break

            valid_frame += 1

            if valid_frame >= 40 *24:
                break

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            img_input = transform({"image": img})["image"]

            if img_inputs is None:
                img_inputs = img_input[None]
            else:
                img_inputs = np.concatenate((img_inputs, img_input[None]), axis=0)

        img_num, c, h, w = img_inputs.shape
        predictions = np.zeros((img_num, h, w)).astype(np.float32)
        count = np.zeros((img_num, 1, 1)).astype(np.float32)

        done = False
        for i in range(0, img_num, seq_len - overlap * 2):
            if i+interval*seq_len >= img_num:
                done = True
                sample = img_inputs[img_num-interval*seq_len:img_num:interval]
            else:
                sample = img_inputs[i:i+interval*seq_len:interval]

            # compute
            with torch.no_grad():
                # sample = np.stack(img_inputs, axis=0)
                sample = torch.from_numpy(sample).to(device)

                if optimize == True and device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()

                sample = sample.unsqueeze(0)

                prediction = model.forward(sample).squeeze(0)
                prediction = (
                    prediction
                        .squeeze()
                        .cpu()
                        .numpy()
                )

            if i == 0:
                predictions[i:i+interval*(seq_len-overlap):interval] = predictions[i:i+interval*(seq_len-overlap):interval] + prediction[:seq_len-overlap]
                count[i:i+interval*(seq_len-overlap):interval] = count[i:i+interval*(seq_len-overlap):interval] + 1
            elif i == img_num - interval * (seq_len - 1) - 1:
                predictions[i+interval*overlap:i+interval*seq_len:interval] = predictions[i+interval*overlap:i+interval*seq_len:interval] + prediction[overlap:]
                count[i+interval*overlap:i+interval*seq_len:interval] = count[i+interval*overlap:i+interval*seq_len:interval] + 1
            else:
                predictions[i+interval*overlap:i+interval*(seq_len-overlap):interval] = predictions[i+interval*overlap:i+interval*(seq_len-overlap):interval] + prediction[overlap:seq_len-overlap]
                count[i+interval*overlap:i+interval*(seq_len-overlap):interval] = count[i+interval*overlap:i+interval*(seq_len-overlap):interval] + 1

            if done:
                break

        predictions = predictions / count

        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 255
        predictions = predictions.astype(np.uint8)

        videoWriter = cv2.VideoWriter(filename, fourcc, fps, (img.shape[1], img.shape[0]))
        colormap = plt.get_cmap('inferno')
        for i in range(predictions.shape[0]):
            heatmap = (colormap(predictions[i]) * 2 ** 8).astype(np.uint8)[:, :, :3]
            prediction = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            prediction = cv2.resize(prediction, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            videoWriter.write(prediction)

        videoWriter.release()
        del videoWriter

    print("finished")


if __name__ == "__main__":

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input_video", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--checkpoint",
                        default='checkpoints/vita-hybrid-intval=3.pth',
                        help='checkpoint')

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.checkpoint,
        args.optimize,
    )
