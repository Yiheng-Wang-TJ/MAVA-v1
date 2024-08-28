import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import argparse
import logging
import yaml
import importlib
from sentence_transformers import SentenceTransformer


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_module(module_path):
    module_parts = module_path.split('.')
    module = importlib.import_module('.'.join(module_parts[:-1]))
    class_ = getattr(module, module_parts[-1])
    return class_

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class MLVU(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            # with open(os.path.join(data_dir, v[0]), 'r') as f:
            with open('/root/MLVU/json/2_needle.json', 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }



def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    if pred==gt_option:
        flag=True

    return flag


def main(config_path, args):
    config = load_config(config_path)
    model_clip = SentenceTransformer(config['model_clip'])
    storage_path = config['storage_path']
    modules_config = config['modules']

    result_logger = logging.getLogger('result_logger')
    result_logger.setLevel(logging.INFO)
    result_handler = logging.FileHandler(f'result.log')
    result_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    result_logger.addHandler(result_handler)

    other_logger = logging.getLogger('other_logger')
    other_logger.setLevel(logging.INFO)
    other_handler = logging.FileHandler(f'other.log')
    other_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    other_logger.addHandler(other_handler)

    # get modules
    active_modules = {}
    for module_name, module_path in modules_config.items():
        module_class = load_module(module_path)
        active_modules[module_name] = module_class
        
    # dataset configuration and builder
    data_list = {
        "needle": ("2_needle.json", f"/root/MLVU/MLVU/video/2_needle", "video"),
    }
    data_dir = ""
    save_path = f"interval{args.interval}-compress{args.compression_rate}-retrieval{args.retrieval_number}-v2"
    dataset = MLVU(data_dir, data_list)
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    # to support restart from beackpoint
    if os.path.exists(f"{save_path}.json"):
        with open(f"{save_path}.json", "r") as f:
            json_data = json.load(f)
        res_list = json_data['res_list']
        acc_dict = json_data['acc_dict']
        for key in acc_dict.keys():
            correct += acc_dict[key][0] 
            total += acc_dict[key][1]

    print("res list", res_list)
    # # **************************** find out why ***********************************************
    # questions = [example['question'] for example in dataset]       
    # questions_set = set(questions)
    # print(f"there are {len(questions)} questions, while {len(questions_set)} questions are unique")
    
    # index_map = {}
    # duplicates = {}
    # for idx, item in enumerate(questions):
    #     if item in index_map:
    #         if item not in duplicates:
    #             duplicates[item] = [index_map[item]]
    #         duplicates[item].append(idx)
    #     else:
    #         index_map[item] = idx
    # print(duplicates)
    
    # *********************************************************************************
    # run the evaluation
    for example in tqdm(dataset):              
        if (example['question'], example['video']) in [(item['question'], item['video']) for item in res_list]:
            # print(f"pass question {example['question']} on video {example['video']}")
            continue
        else:
            print(f"running on question {example['question']} on video {example['video']}")
            
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        task_id = 100

        key = "AIzaSyA2Ez04AJoq-2ZsAuGkw1gyoltmT_OWAEk"

        # for inference
        video_path = example["video"]
        query = example["question"][9:]
        prompt = "You will be given a question about a video and four possible answer options. You will be provided frames from the video sampled across the video. You must output the final answer in the format '(X)' where X is the correct choice in A,B,C,D. DO NOT OUTPUT with the full answer text or any other words."
        
        interval = args.interval
        compression_rate = args.compression_rate
        retrieval_number = args.retrieval_number
        if 'compression' in active_modules:
            print("start compression")
            compression_module = active_modules['compression'](video_path, interval, compression_rate, storage_path, model_clip)
            video_frames, frames_index = compression_module.sample(other_logger)
            compressed_frames, index_node = compression_module.compress(frames_index, other_logger)
        else:
            raise ValueError("Compression module is required but not enabled.")

        if 'encoding' in active_modules:
            print("start encoding")
            encoding_module = active_modules['encoding'](model_clip, compressed_frames, storage_path)
            img_emb = encoding_module.encoding(other_logger)
        else:
            raise ValueError("Encoding module is required but not enabled.")

        if 'storage' in active_modules:
            storage_module = active_modules['storage'](img_emb)
            data = storage_module.storage()
        else:
            raise ValueError("Storage module is required but not enabled.")

        if 'retrieval' in active_modules:
            print("start retrieval")
            retrieval_module = active_modules['retrieval'](model_clip, query, data, compressed_frames, retrieval_number, index_node)
            retrieved_data = retrieval_module.retrieval(other_logger)
        else:
            raise ValueError("Retrieval module is required but not enabled.")

        if 'conversation' in active_modules:
            print("start conversation")
            conversation_module = active_modules['conversation'](retrieved_data, query, prompt, storage_path)
            response = conversation_module.generate_response(other_logger, result_logger, key)
        else:
            response = retrieved_data

        result_logger.info('Final result: %s', response)
    
        gt = example['answer']
        res_list.append({
            'pred': response,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video':example['video']
        })
        if check_ans(pred=response, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

        with open(f"{save_path}.json", "w") as f:
            json.dump({
                "acc_dict": acc_dict,
                "res_list": res_list
            }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference pipeline")
    parser.add_argument('--config', type=str, default='/root/Memory-Augmented-Video-Agent-main/config/Config.yaml', help="Path to the configuration file")
    parser.add_argument('--interval', type=int, default=6,
                        help="Interval for frame sampling")
    parser.add_argument('--compression_rate', type=int, default=2,
                        help="Compression rate for reducing frames")
    parser.add_argument('--retrieval_number', type=int, default=10,
                        help="Number of retrievals")
    args = parser.parse_args()
    main(args.config, args)
