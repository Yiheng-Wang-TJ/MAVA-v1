import time
import torch
from mava.retrieval.base_retrieval import BaseRetrieval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HierarchicalRetrieval(BaseRetrieval):
    def __init__(self, model_clip, query, data, compressed_frames, retrieval_number, index_node):
        self.model_clip = model_clip
        self.query = query
        self.data = data
        self.compressed_frames = compressed_frames
        self.retrieval_number = retrieval_number
        self.index_node = index_node

    def calculate_centroids(self):
        num_levels = self.index_node.shape[1]
        centroids = []

        for level in range(num_levels):
            unique_labels = np.unique(self.index_node[:, level])
            level_centroids = []

            for label in unique_labels:
                cluster_points_indices = np.where(self.index_node[:, level] == label)[0]
                cluster_points = self.data[cluster_points_indices]
                centroid = cluster_points.mean(axis=0)
                level_centroids.append((label, centroid))

            centroids.append(level_centroids)

        return centroids

    def retrieval(self, other_logger):
        start = time.perf_counter()

        num_levels = self.index_node.shape[1]
        results = []
        centroids = self.calculate_centroids()

        with torch.no_grad():
            query_emb = self.model_clip.encode(
                [self.query],
                convert_to_tensor=True,
                show_progress_bar=False
            ).cpu().numpy()

        accumulated_results = []
        current_indices = np.arange(self.index_node.shape[0])

        for level in range(num_levels - 1, -1, -1):
            level_centroids = centroids[level]

            similarities = []
            for label, centroid in level_centroids:
                centroid_cpu = centroid.cpu().numpy()
                similarity = cosine_similarity(centroid_cpu.reshape(1, -1), query_emb.reshape(1, -1)).flatten()[0]
                similarities.append((label, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)

            label_index = 0
            while len(accumulated_results) < self.retrieval_number and label_index < len(similarities):
                most_similar_label, _ = similarities[label_index]

                cluster_indices = current_indices[np.where(self.index_node[current_indices, level] == most_similar_label)[0]]

                accumulated_results = cluster_indices.tolist() + accumulated_results

                current_indices = np.array(cluster_indices)

                label_index += 1

                if level == 0 and len(accumulated_results) >= self.retrieval_number:
                    break

            if level == 0 and len(accumulated_results) < self.retrieval_number:
                continue

            if len(accumulated_results) >= self.retrieval_number:
                break

        results = accumulated_results[:self.retrieval_number]
        select_indexs = [self.compressed_frames[i] for i in results]

        end = time.perf_counter()
        print('retrieval cost: ', (end - start) * 1000)
        other_logger.info('retrieval cost:%s ms', (end - start) * 1000)

        print(select_indexs)
        other_logger.info('select_indexs:%s', select_indexs)

        return select_indexs
