"""InfiniPot offline evaluation for all existing offline datasets."""
from video_qa.base import work
from video_qa.infinipot_vqa import InfiniPotVQA


class InfiniPotOfflineVQA(InfiniPotVQA):
    pass


if __name__ == '__main__':
    work(InfiniPotOfflineVQA, method='infinipot')
