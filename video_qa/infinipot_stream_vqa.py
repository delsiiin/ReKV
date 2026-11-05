"""InfiniPot visible-prefix evaluation for RVS Ego/Movie."""
from video_qa.base import work
from video_qa.infinipot_vqa import InfiniPotVQA


class InfiniPotStreamVQA(InfiniPotVQA):
    streaming = True


if __name__ == '__main__':
    work(InfiniPotStreamVQA, method='infinipot')
