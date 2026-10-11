"""Sample selection shared by the evaluation launcher and inference workers."""
import argparse
import json


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def load_samples(anno_path, max_samples=0):
    """Select the first max_samples video records; zero means all records."""
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    with open(anno_path) as annotation_file:
        samples = json.load(annotation_file)
    return samples[:max_samples] if max_samples else samples


def split_samples(samples, num_chunks):
    """Split records into exactly num_chunks nonempty, contiguous chunks."""
    if not 1 <= num_chunks <= len(samples):
        raise ValueError("num_chunks must be between 1 and the number of selected samples")
    size, remainder = divmod(len(samples), num_chunks)
    chunks = []
    start = 0
    for index in range(num_chunks):
        end = start + size + (index < remainder)
        chunks.append(samples[start:end])
        start = end
    return chunks
