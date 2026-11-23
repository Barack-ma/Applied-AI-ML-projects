import numpy as np

def set_seed(seed):
    np.random.seed(seed)

def make_embedding_matrix(num_tokens, embedding_dim):
    rng = np.random.default_rng(412)
    return rng.standard_normal((num_tokens, embedding_dim))
