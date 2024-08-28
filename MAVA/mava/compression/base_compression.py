import time
import subprocess
import cv2
from sentence_transformers import util
from PIL import Image
import os
import torch

class BaseCompression:
    def __init__(self, video, interval, compression_rate, storage_path, model_clip):
        self.video = video
        self.interval = interval
        self.compression_rate = compression_rate
        self.storage_path = storage_path
        self.model_clip = model_clip
        
    def sample(self, other_logger):
        del_list = os.listdir(self.storage_path)
        for f in del_list:
            file_path = os.path.join(self.storage_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        start = time.perf_counter()

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        ffmpeg_command = [
            'ffmpeg',
            '-i', self.video,
            '-vf', f'select=not(mod(n\\,{self.interval}))',
            '-vsync', 'vfr',
            f'{self.storage_path}/frame_%06d.png'
        ]

        subprocess.run(ffmpeg_command, check=True)

        video_frames = sorted([f for f in os.listdir(self.storage_path) if f.endswith('.png')])
        for i, frame in enumerate(video_frames):
            new_name = f'{i:06d}.png'
            os.rename(os.path.join(self.storage_path, frame), os.path.join(self.storage_path, new_name))

        end = time.perf_counter()
        print(end - start)
        
        video_frames = sorted([os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path) if f.endswith('.png')])
        frames_index = [int(os.path.splitext(os.path.basename(frame))[0]) for frame in video_frames]
        
        other_logger.info('processing:%s s', self.video)
        other_logger.info('video_frames len:%s', len(video_frames))
        other_logger.info('ffmpeg cost:%s s', end - start)
        return video_frames, frames_index
    
    # **************************************Replace sample *********************************************************
    # def sample(self, other_logger): 
    #     del_list = os.listdir(self.storage_path)
    #     for f in del_list:
    #         file_path = os.path.join(self.storage_path, f)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
        
    #     frame_index = 0
    #     start = time.perf_counter()
    #     video_cv = cv2.VideoCapture(self.video)
    #     video_frames = []
    #     frames_index = []

    #     while video_cv.isOpened():
    #         success, frame = video_cv.read()
    #         if not success:
    #             break
    #         if frame_index % self.interval == 0:
    #             save_path = "{}/{:>06d}.png".format(self.storage_path, frame_index)
    #             video_frames.append(save_path)
    #             frames_index.append(frame_index)
    #             cv2.imwrite(save_path, frame)
    #         frame_index += 1

    #     end = time.perf_counter()
    #     print("convert video to frames cost", end - start)

    #     other_logger.info('opencv cost:%s s', end - start)
    #     other_logger.info('video_frames len:%s', len(video_frames))

    #     return video_frames, frames_index

    def compress(self, frames_index, other_logger):
        compress_frames = frames_index
        features = []

        start = time.perf_counter()
        for frame in compress_frames:
            frame = "{}/{:>06d}.png".format(self.storage_path, frame)
            img_emb = self.model_clip.encode(Image.open(frame))
            features.append(img_emb)

        number = int(len(compress_frames) / self.compression_rate)
        end = time.perf_counter()
        print("encode features cost", end - start)
        other_logger.info('encode features cost:%s', end - start)

        start = time.perf_counter()
        cos_scores = []
        for i in range(0, len(compress_frames) - 1):
            cos_scores.append(util.cos_sim(features[i], features[i + 1]))
        while len(compress_frames) > number:
            value, index = torch.max(torch.tensor(cos_scores), 0)
            compress_frames.pop(index+1)
            features.pop(index+1)
            # ..., index, index+1, index+2, ... ---> ..., index, index+1(the original index+2), ...
            if index+2 <= len(features)-1:
                cos_scores.pop(index+1)  # cosine similarity between original index+1 feature and originl index+2 feature
                cos_scores.pop(index)    # cosine similarity between original index feature and original index+1 feature
                cos_scores.insert(index, util.cos_sim(features[index], features[index + 1]))
            else:
                cos_scores.pop(index)
        end = time.perf_counter()
        print("compress cost", end - start)
        other_logger.info('compress_cost:%s', end - start)
        other_logger.info('compress_frames:%s', len(compress_frames))
        return compress_frames, compress_frames
        
    # **************************************Replace compress*********************************************************
    # def compress(self, frames_index, other_logger):
    #     compress_frames = frames_index
    #     features = []

    #     start = time.perf_counter()
        
    #     for frame in compress_frames:
    #         frame = "{}/{:>06d}.png".format(self.storage_path, frame)
    #         img_emb = self.model_clip.encode(Image.open(frame))
    #         features.append(img_emb)

    #     number = int(len(compress_frames) / self.compression_rate)

    #     end = time.perf_counter()
    #     print("encode features cost", end - start)

    #     start = time.perf_counter()
    #     while len(compress_frames) > number:
    #         index = 9999
    #         max = -9999

    #         for i in range(0, len(compress_frames) - 1):
    #             cos_scores = util.cos_sim(features[i], features[i + 1])
    #             if cos_scores > max:
    #                 max = cos_scores
    #                 index = i

    #         compress_frames.pop(index + 1)
    #         features.pop(index + 1)
    #     end = time.perf_counter()
    #     print("compress cost", end - start)
        
    #     other_logger.info('compress_frames:%s', len(compress_frames))

    #     return compress_frames, compress_frames
