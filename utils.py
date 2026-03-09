import numpy as np

rng = np.random.default_rng(seed=42)

def sample_groupwise_rels(args, low=0.0, high=1.0, g_label=0, r=3):
    rels = rng.uniform(low=low, high=high, size=(int(args.m), int(r))) # relevances
    gs = np.array([g_label]*(int(r))).repeat(int(args.m),axis=0).reshape(int(args.m), int(r)).astype(np.float32)
    return rels, gs


def _position_weights(m):
    return 1.0 / np.log(np.arange(1, m + 1, dtype=np.float32) + 1.0)


def unfairness(b, rel, g):
    b = np.squeeze(np.asarray(b))
    rel = np.asarray(rel)
    g = np.asarray(g)
    assert b.ndim == 2 and b.shape[0] == b.shape[1]
    m = rel.shape[0]
    w = _position_weights(m)[:, None]
    _, counts = np.unique(g, return_counts=True)
    if counts.shape[0] != 2:
        return np.array(0.0), None
    n_g1, n_g2 = counts
    rel = rel[None, :]
    g = g[None, :]
    w_g1 = w * (1 - g) # 0 is for group 1
    w_g2 = w * g # 1 is for group 2
    rel = (rel * b)
    util_g1 = np.tensordot(rel, w_g1, axes=2) / n_g1
    util_g2 = np.tensordot(rel, w_g2, axes=2) / n_g2
    # normalize by group size
    unfairness = np.abs(util_g1 - util_g2)
    unfairness = float(unfairness)
    return unfairness


def utility(b, rel):
    b = np.squeeze(np.asarray(b))
    rel = np.asarray(rel)
    assert b.ndim == 2 and b.shape[0] == b.shape[1]
    m = rel.shape[0]
    w = np.broadcast_to(_position_weights(m)[:, None], (m, m))
    rel = rel[None, :]
    # expand w and rel to 8x8 and multiply to get util
    rel = (rel * b)
    util = np.tensordot(rel, w, axes=2)
    return float(util)


# minimizing unfairness and maximizing utility
def utility_and_unfairness(b, rel, g, alpha=0.5):
    return (1-alpha)*unfairness(b, rel, g) - alpha*utility(b, rel)



def unfairness_greedy(b, rel, g):
    b = np.asarray(b, dtype=int).reshape(-1)
    rel = np.asarray(rel)
    g = np.asarray(g)
    m = rel.shape[0]
    w = _position_weights(m)
    idx_g1, idx_g2 = np.nonzero(g == 0)[0], np.nonzero(g == 1)[0]
    n_g1, n_g2 = len(idx_g1), len(idx_g2)
    if n_g1 == 0 or n_g2 == 0:
        return np.array([0.0])
    # clip indices to what is contained in b
    idx_g1_trunc = idx_g1[np.isin(idx_g1, b)]
    idx_g2_trunc = idx_g2[np.isin(idx_g2, b)]
    n_g1_trunc, n_g2_trunc = len(idx_g1_trunc), len(idx_g2_trunc)
    # if n_g1_trunc == 0 or n_g2_trunc == 0:
        # return 0
    # slice weights by position in b
    w_g1 = w[np.nonzero(np.isin(b, idx_g1_trunc))[0]]
    w_g2 = w[np.nonzero(np.isin(b, idx_g2_trunc))[0]]
    rel_g1 = rel[b][(g[b] == 0)]
    rel_g2 = rel[b][(g[b] == 1)]
    util_g1 = float(np.dot(rel_g1, w_g1)) / n_g1 # << divide by sum of docs in that group
    util_g2 = float(np.dot(rel_g2, w_g2)) / n_g2 # << divide by sum of docs in that group
    # util_g1 = np.dot(rel_g1, w_g1)/(n_g1_trunc + 1) # << divide by sum of docs in that group
    # util_g2 = np.dot(rel_g2, w_g2)/(n_g2_trunc + 1) # << divide by sum of docs in that group
    # util_g1 = np.dot(rel_g1, w_g1)/((n_g1-n_g1_trunc) + 1) # << divide by sum of docs in that group
    # util_g2 = np.dot(rel_g2, w_g2)/((n_g2-n_g2_trunc) + 1) # << divide by sum of docs in that group
    util_g1 = np.nan_to_num(util_g1, nan=0.0)
    util_g2 = np.nan_to_num(util_g2, nan=0.0)
    # normalize by group size
    unfairness = np.abs(util_g1 - util_g2)#*(n_g1_trunc+n_g2_trunc)/(n_g1+n_g2) #extensive search showed that nonrescaled truncated unfairness calculation works better
    return float(unfairness)


def utility_greedy(b, rel):
    b = np.asarray(b, dtype=int).reshape(-1)
    rel = np.asarray(rel)
    m = rel.shape[0]
    w = _position_weights(m)[:b.shape[-1]]
    rel = rel[b]
    util = np.dot(rel, w.flatten())
    return float(util)


def utility_and_unfairness_greedy(b, rel, g, alpha=0.5):
    return (1-alpha)*unfairness_greedy(b, rel, g) - alpha*utility_greedy(b, rel)


def make_permutation_matrix(b):
    # permutation matrix P_b with column representation: p_{ij} = 1 if j = b(i)
    # shs: cols are docs, rows are positions
    return np.eye(b.shape[-1])[b]