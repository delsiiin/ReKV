"""Method-specific video state and output metadata, separate from QA traversal."""
import json
import math

import torch


class RetrievalMetadata:
    def __init__(self, *args, retrieve_size=64, chunk_size=1, **kwargs):
        if not 1 <= chunk_size <= retrieve_size:
            raise ValueError('Require 1 <= chunk_size <= retrieve_size')
        self.retrieve_size, self.chunk_size = retrieve_size, chunk_size
        super().__init__(*args, record_metadata=dict(retrieve_size=retrieve_size, chunk_size=chunk_size),
                         **kwargs)

    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0

        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1


class ReKVLifecycle(RetrievalMetadata):
    def start_video(self):
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

    def consume_frame(self, frame, timestamp):
        self.qa_model.encode_video(torch.from_numpy(frame).unsqueeze(0))


class SuPrKVLifecycle(RetrievalMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_stats = []

    def start_video(self):
        self.qa_model.reset()

    def consume_frame(self, frame, timestamp):
        self.qa_model.append_frame(frame, timestamp)

    def finish_encoding(self):
        self.qa_model.close_segment()

    def finish_video(self, video_sample):
        self.video_stats.append(dict(video_id=video_sample['video_id'],
                                     config=self.qa_model.config.__dict__,
                                     memory=self.qa_model.memory_stats()))

    def analyze(self, debug=False):
        super().analyze(debug)
        path = f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.stats.jsonl'
        with open(path, 'w', encoding='utf-8') as output:
            for stats in self.video_stats:
                output.write(json.dumps(stats, ensure_ascii=False) + '\n')


class NativeLifecycle:
    def start_video(self):
        self.qa_model.reset()

    def consume_frame(self, frame, timestamp):
        self.qa_model.append_frame(frame, timestamp)
