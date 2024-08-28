import time
import torch
from PIL import Image

class BaseEncoding:
    def __init__(self, model_clip, compressed_frames, storage_path):
        self.model_clip = model_clip
        self.compressed_frames = compressed_frames
        self.storage_path = storage_path

    def encoding(self, other_logger):
        start = time.perf_counter()
        video_frames = []
        for frame_index in self.compressed_frames:
            video_frames.append(
                "{}/{:>06d}.png".format(self.storage_path, frame_index))
        with torch.no_grad():
            img_emb = self.model_clip.encode(
                [Image.open(filepath) for filepath in video_frames],
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=True
            )
        end = time.perf_counter()
        print('clip encoding cost: ', end - start)
        other_logger.info('clip encoding cost:%s s', end - start)
        return img_emb
