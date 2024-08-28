import time
from sentence_transformers import util
import torch

class BaseRetrieval:
    def __init__(self, model_clip, query, data, compressed_frames, retrieval_number, index_node):
        self.model_clip = model_clip
        self.query = query
        self.data = data
        self.compressed_frames = compressed_frames
        self.retrieval_number = retrieval_number

    def retrieval(self, other_logger):
        select_indexs = []
        start_position = 0

        with torch.no_grad():
            query_emb = self.model_clip.encode(
                [self.query],
                convert_to_tensor=True,
                show_progress_bar=False
            )

        start = time.perf_counter()
        hits = util.semantic_search(query_emb, self.data, top_k=self.retrieval_number)[0]
        end = time.perf_counter()
        print('retrieval cost: ', (end - start) * 1000)
        other_logger.info('retrieval cost:%s ms', (end - start) * 1000)
        for hit in range(start_position, self.retrieval_number):
            select_indexs.append(self.compressed_frames[hits[hit]['corpus_id']])
        select_indexs.sort()

        print(select_indexs)
        other_logger.info('select_indexs:%s', select_indexs)
        return select_indexs
