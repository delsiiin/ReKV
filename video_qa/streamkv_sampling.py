"""StreamKV reference frame selection, including 8-FPS image directories."""
import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu


class StreamKVVideoSource:
    def load_video(self, video_path):
        if os.path.isdir(video_path):
            img_files = sorted(
                [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            )
            if len(img_files) == 0:
                raise FileNotFoundError(f"No jpg/png images found in {video_path}")
            imgs = []
            for filename in img_files:
                img = Image.open(os.path.join(video_path, filename)).convert('RGB')
                imgs.append(np.array(img))
            video = np.stack(imgs, axis=0)
            if  self.sample_fps < 8:
                frame_idx = np.arange(0, len(imgs), int(8 // self.sample_fps))
                video = video[frame_idx]
        elif video_path.endswith('.npy'):
            video = np.load(video_path)
            assert self.sample_fps <= 1
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        return video

