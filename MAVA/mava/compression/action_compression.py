import subprocess
import os
import time
from mava.compression.base_compression import BaseCompression
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
from PIL import Image
from sentence_transformers import util

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

# use Approx NN to find first neighbor if samples more than ANN_THRESHOLD
ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine', use_tw_finch=False):
    s = mat.shape[0]
    
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        if use_tw_finch:
            loc = mat[:, -1]
            mat = mat[:, :-1]            
            loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2)            
            
        else:
            loc_dist = 1.            

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        orig_dist = orig_dist * loc_dist
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if use_tw_finch:
            print(f'Since the video is larger than {ANN_THRESHOLD} samples, we cannot compute all distances. Instead FINCH will be used')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
            )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)   
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a

def req_numclust(c, data, req_clust, distance, use_tw_finch=False):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', tw_finch=True, ensure_early_exit=False, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param tw_finch: Run TW_FINCH on video data.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    if tw_finch:
        n_frames = data.shape[0]
        time_index = (np.arange(n_frames) + 1.) / n_frames       
        data = np.concatenate([data, time_index[..., np.newaxis]], axis=1)
        ensure_early_exit = False
        verbose = False

    # Cast input data to float32
    data = data.astype(np.float32)
    
    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance=distance, use_tw_finch=tw_finch)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:        
        adj, orig_dist = clust_rank(mat, initial_rank, distance=distance, use_tw_finch=tw_finch)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_tw_finch=tw_finch)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c


class FINCHCompression(BaseCompression):
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

        other_logger.info('ffmpeg cost:%s s', end - start)

        video_frames = sorted(
            [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path) if f.endswith('.png')])
        frames_index = [int(os.path.splitext(os.path.basename(frame))[0]) for frame in video_frames]

        other_logger.info('video_frames len:%s', len(video_frames))

        return video_frames, frames_index

    def compress(self, frames_index, other_logger):
        compress_frames = frames_index
        features = []

        for frame in compress_frames:
            frame = "{}/{:>06d}.png".format(self.storage_path, frame)
            img_emb = self.model_clip.encode(Image.open(frame))
            features.append(img_emb)

        features = np.array(features)

        start = time.perf_counter()

        c, num_clust, req_c = FINCH(features, initial_rank=None, req_clust=None, distance='cosine',
                                    ensure_early_exit=True, verbose=True)

        print(num_clust)
        print(c)

        column_index = 2 if c.shape[1] > 2 else -1
        column_data = c[:, column_index]

        unique_labels = np.unique(column_data)
        label_indices = {label: np.where(column_data == label)[0] for label in unique_labels}

        if c.shape[1] > 2:
            index_node = c[:, 2:]
        else:
            index_node = c[:, -1]

        end = time.perf_counter()
        print(end - start)

        for label, indices in label_indices.items():
            if len(indices) <= self.compression_rate:
                continue
            number = int(len(indices) / self.compression_rate)

            filtered_rows = index_node[index_node[:, 0] == label]

            features = []

            for frame in indices:
                frame = "{}/{:>06d}.png".format(self.storage_path, frame)
                img_emb = self.model_clip.encode(Image.open(frame))
                features.append(img_emb)

            while len(indices) > number:
                index = 9999
                max = -9999

                for i in range(0, len(indices) - 1):
                    cos_scores = util.cos_sim(features[i], features[i + 1])
                    if cos_scores > max:
                        max = cos_scores
                        index = i

                indices = np.delete(indices, index + 1)
                features.pop(index + 1)
                filtered_rows = np.delete(filtered_rows, index + 1, axis=0)

            label_indices[label] = indices

            index_node = index_node[index_node[:, 0] != label]
            index_node = np.vstack([index_node, filtered_rows])

        indices_list = []
        for indices in label_indices.values():
            indices_list.extend(indices.tolist())

        return indices_list, index_node