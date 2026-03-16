import argparse
from functools import partial
import time
import numpy as np
import multiprocessing as mp
import pandas as pd

from utils import unfairness_from_positions
from itertools import combinations


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6, help='Dimension of vector to sort')
    parser.add_argument('--r', type=int, default=3, help='Number of Group A documents')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)


def fixed_rel_fair_ranking(rel_A, rel_B, r, s):
    """
    Enumerate all C(N, n1) ways to assign positions to G1 and find the best.

    Since all docs within a group are identical, we only need to decide
    *which positions* go to G1 — the rest go to G2.

    Parameters
    ----------
    rel_A, rel_B : scalar qualities for group A and group B
    r, s : number of documents in each group

    Returns
    -------
    best_positions_g1 : frozenset of positions assigned to G1
    best_score        : unfairness value
    """
    m = r + s
    all_positions = np.arange(m)
    best_score, best_positions = float("inf"), None

    g1_pos_combs = list(combinations(range(m), r))
    num_cores = mp.cpu_count()
    partial_function = partial(unfairness_from_positions, relA_scalar=rel_A, relB_scalar=rel_B, all_positions=all_positions)

    with mp.Pool(processes=num_cores) as pool:
        # scores is a list of return values from the workers
        scores = pool.map(partial_function, g1_pos_combs)

    min_score_index = np.argmin(scores)
    best_positions, best_score = g1_pos_combs[min_score_index], scores[min_score_index]
    # for g1_pos in combinations(range(m), r):
    #     score = unfairness_from_positions(rel_A, rel_B, g1_pos, all_positions)
    #     if score < best_score:
    #         best_score = score
    #         best_positions = g1_pos

    return best_positions, best_score



if __name__ == "__main__":
    args = _parse_args(None)
    ratios = np.logspace(0.0, 0.3, 12)
    # ratios = np.array([1.5])
    relBs = [1]*len(ratios)  # Keep relBs constant
    relAs = ratios * np.array(relBs)  # Calculate relAs based on the ratios and constant relBs
    r = args.r
    
    
    # relBs = np.array([0.6])
    # relAs = np.array([0.9])
    # ratios = relAs / relBs
    data_list = list()

    for relA, relB, ratio in zip(relAs, relBs, ratios):

        print(f"ratio={ratio} Group A rel={relA}, Group B rel={relB}, r={r}")

        # rels = np.array([relA]*r + [relB]*(args.k - r))  # dummy rels since exact_fair_ranking only needs scalar relA and relB
        # gs = np.array([0]*r + [1]*(args.k - r))  # dummy
        # rels = np.tile(rels, (1, 1))
        # gs = np.tile(gs, (1, 1))
        # results = get_util_and_unfairness_on_samples(args, rels, gs, alpha_list=[])
        # print(rels[0][np.array(results['min_V_perm'][0])])
        # gA_pos = gs[0][np.array(results['min_V_perm'][0])]
        # print(gA_pos)
        # print(results['min_V'])
        # pattern = "".join("A" if p == 0 else "B" for p in gA_pos)
        # print(pattern)

        start_time = time.perf_counter()
        gA_pos, unfairness = fixed_rel_fair_ranking(relA, relB, r, args.k - r)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        pattern = "".join("A" if k in gA_pos else "B" for k in range(args.k))
        print(f"Best positions for G1: {gA_pos}")
        print(f"Unfairness: {unfairness}")
        print(f"Pattern: {pattern}")
        
        print(f"Function executed in {elapsed_time:.4f} seconds")
        data_list.append((relA, relB, ratio, r, unfairness, pattern, elapsed_time))

    column_names = ['relA', 'relB', 'ratio', 'r', 'unfairness', 'pattern', 'elapsed_time']

    # Create the DataFrame
    df = pd.DataFrame(data_list, columns=column_names)
    df.to_csv(f'./exp_log/exact_sol_k_{args.k}.csv', index=False)





