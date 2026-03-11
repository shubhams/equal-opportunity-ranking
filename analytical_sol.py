import argparse
import time
import numpy as np

from utils import unfairness_from_positions
from itertools import combinations


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6, help='Dimension of vector to sort')
    parser.add_argument('--m', type=int, default=100, help='Number of documents to sample from the bag of documents')
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

    for g1_pos in combinations(range(m), r):
        score = unfairness_from_positions(rel_A, rel_B, g1_pos, all_positions)
        if score < best_score:
            best_score = score
            best_positions = g1_pos

    return best_positions, best_score



if __name__ == "__main__":
    # Example usage
    class Args:
        k = 100  # dimension of vector to sort
        m = 2
    args = Args()
    ratios = np.logspace(0.0, 0.3, 2)
    # ratios = np.array([1.5])
    relBs = [1]*len(ratios)  # Keep relBs constant
    relAs = ratios * np.array(relBs)  # Calculate relAs based on the ratios and constant relBs
    
    
    # relBs = np.array([0.6])
    # relAs = np.array([0.9])
    # ratios = relAs / relBs

    for relA, relB, ratio in zip(relAs, relBs, ratios):
        # for r in range(1, args.k):
        # for r in range(3, 4):
        for r in range(5, 6):
            print(f"relB: {relB}, ratio={ratio} Group A rel={relA}, Group B rel={relB}, r={r}")

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




