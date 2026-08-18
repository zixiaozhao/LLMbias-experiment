from .pairwise import PairwiseModel


class Qwen3(PairwiseModel):
    SAMPLING_OVERRIDES = {"min_p": 0}
