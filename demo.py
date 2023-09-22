#
# Copyright (C) 2023, NTU
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  ke.xian@ntu.edu.sg or xianke1991@gmail.com
#

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
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


def run_video(input_path, output_path, model, checkpoint,  optimize=True):

    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # save video with 12 fps
    fps = 12

    # load network
    net_w = net_h = 384
    model = DP(model)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)

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


def run_imgs(input_path, output_path, model, checkpoint,  optimize=True):

    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # save video with 12 fps
    fps = 12

    # load network
    net_w = net_h = 384
    model = DP(model)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)

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
    img_names = sorted(os.listdir(input_path))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    img_inputs = None
    predictions = None
    for ind, img_name in enumerate(img_names):
        img_name = os.path.join(input_path, img_name)

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        image = cv2.imread(img_name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        img_input = transform({"image": img})["image"]

        if img_inputs is None:
            img_inputs = img_input[None]
        else:
            img_inputs = np.concatenate((img_inputs, img_input[None]), axis=0)

    overlap = 1  # 0
    interval = 1
    seq_len = 4
    shift = 0

    img_num, c, h, w = img_inputs.shape
    predictions = np.zeros((img_num, h, w)).astype(np.float32)
    count = np.zeros((img_num, 1, 1)).astype(np.float32)

    done = False
    for i in range(0, img_num, seq_len - overlap * 2):
        if i + interval * seq_len >= img_num:
            done = True
            sample = img_inputs[img_num - interval * seq_len:img_num:interval]
        else:
            sample = img_inputs[i:i + interval * seq_len:interval]

        # compute
        with torch.no_grad():
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
            predictions[i:i + interval * (seq_len - overlap):interval] = predictions[i:i + interval * (seq_len - overlap):interval] + prediction[:seq_len - overlap]
            count[i:i + interval * (seq_len - overlap):interval] = count[i:i + interval * (seq_len - overlap):interval] + 1
        elif i == img_num - interval * (seq_len - 1) - 1:
            predictions[i + interval * overlap:i + interval * seq_len:interval] = predictions[i + interval * overlap:i + interval * seq_len:interval] + prediction[overlap:]
            count[i + interval * overlap:i + interval * seq_len:interval] = count[i + interval * overlap:i + interval * seq_len:interval] + 1
        else:
            predictions[i + interval * overlap:i + interval * (seq_len - overlap):interval] = predictions[i + interval * overlap:i + interval * (seq_len - overlap):interval] + prediction[overlap:seq_len - overlap]
            count[i + interval * overlap:i + interval * (seq_len - overlap):interval] = count[i + interval * overlap:i + interval * (seq_len - overlap):interval] + 1

        if done:
            break

    predictions = predictions / count

    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 255
    predictions = predictions.astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    savename = os.path.join(output_path, os.path.split(input_path)[-1] + '.mp4')
    videoWriter = cv2.VideoWriter(savename, fourcc, fps, (img.shape[1], img.shape[0]))
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

    parser.add_argument(
        "--attn_interval",
        default=2,
        type=int,
        help="1,2,3,4,6",
    )

    parser.add_argument(
        "--format",
        default="video",
        help="model type [video|imgs]",
    )

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    if args.model_type == 'dpt_large':
        model = DPTDepthModel(
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
            attn_interval=args.attn_interval,
        )
        checkpoint = 'checkpoints/vita-large.pth'
    elif args.model_type == 'dpt_hybrid':
        model = DPTDepthModel(
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
            attn_interval=args.attn_interval,
        )
        checkpoint = 'checkpoints/vita-hybrid.pth'

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    if args.format == 'video':
        # run demo on videos
        run_video(
            args.input_path,
            args.output_path,
            model,
            checkpoint,
            args.optimize,
        )
    else:
        # run demo on a image squence
        run_imgs(
            args.input_path,
            args.output_path,
            model,
            checkpoint,
            args.optimize,
        )


