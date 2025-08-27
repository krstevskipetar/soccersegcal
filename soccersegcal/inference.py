"""Run segmentation and camera calibration on a video.

This script loads a trained segmentation network and applies it to each frame of
an input video. The resulting segmentations are used to estimate camera
parameters which are written to a CSV file while an overlay of the predicted
segments is stored as a video.
"""

import csv

import cv2
import numpy as np
import torch

from .train import LitSoccerFieldSegmentation
from .pose import segs2cam
from .sncalib.baseline_cameras import Camera

import fire


def run(video_path, checkpoint, output_video="output.mp4", output_csv="cameras.csv"):
    """Run inference on ``video_path`` using ``checkpoint``.

    Parameters
    ----------
    video_path: str or Path
        Path to input video file.
    checkpoint: str or Path
        Path to the segmentation model checkpoint.
    output_video: str or Path, optional
        Where to store the video with segmentation overlays.
    output_csv: str or Path, optional
        Where to store the estimated camera parameters.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitSoccerFieldSegmentation.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    csv_file = open(output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "pan", "tilt", "roll", "x", "y", "z"])

    frame_idx = 0
    world_scale = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)[None]

        with torch.no_grad():
            segs = torch.sigmoid(model(img))[0].cpu()

        mask = (segs > 0.5).numpy().astype(np.uint8)
        color_mask = np.zeros_like(frame)
        colors = np.array([
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ])
        for idx, color in enumerate(colors[: mask.shape[0]]):
            color_mask[mask[idx] > 0] = color
        overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        writer.write(overlay)

        cam_model = segs2cam(segs, world_scale)
        if cam_model is not None:
            cam_model = cam_model.cpu()
            smallest = min(segs.shape[1:])
            f = smallest / 2 / cam_model.camera_focal.item()
            cam = Camera(width, height)
            cam.from_json_parameters(
                {
                    "position_meters": cam_model.camera_position.detach().numpy() * world_scale,
                    "principal_point": cam.principal_point,
                    "x_focal_length": f,
                    "y_focal_length": f,
                    "pan_degrees": np.rad2deg(cam_model.camera_pan.item()),
                    "tilt_degrees": np.rad2deg(cam_model.camera_tilt.item()),
                    "roll_degrees": np.rad2deg(cam_model.camera_roll.item()),
                }
            )
            x, y, z = cam.position
            csv_writer.writerow([
                frame_idx,
                cam.pan_degrees,
                cam.tilt_degrees,
                cam.roll_degrees,
                x,
                y,
                z,
            ])

        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()

