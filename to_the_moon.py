import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from itertools import permutations, filterfalse, chain
from utils import make_permutation_matrix, utility_and_unfairness, utility, unfairness, utility_and_unfairness_greedy

rng = np.random.default_rng(seed=42)


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=7, help='Dimension of vector to sort')
    parser.add_argument('--m', type=int, default=100, help='Number of documents to sample from the bag of documents')
    parser.add_argument('--exp_path', type=str, default="./exp_log/", help='Where to save all stuff')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--beam_size', type=int, default=2, help='Number of candidates to keep after each iteration of beam search')
    parser.add_argument('--look_ahead', type=int, default=1, help='Number of sorted docs to look ahead for greedy/beam search')
    parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off parameter between utility and unfairness')
    return parser.parse_args(args)


def sample_random_rels_and_gs(args, low=0.0, high=1.0, cate_prob=[0.5,0.5]):
    
    c = np.random.multinomial(args.k, cate_prob, size=args.m)

    rels = rng.uniform(low=low, high=high, size=(args.m, args.k)) # relevances
    # rels = torch.tensor([1.]*args.k).repeat(args.m,1)
    # gs = c.sample(sample_shape=(args.m, args.k)).to(torch.float32) # group memberships
    gs = np.array([0, 1]*(int(args.k/2)) + [0]*(args.k%2)).repeat(args.m,axis=0).astype(np.float32)
    return rels, gs


# TODO: make alphas a list and compute opt tradeoff for all supplied alphas
def opt_ranker(args, rel, g, alpha_list):
    f_util = lambda b: utility(b, rel)
    f_unfair = lambda b: unfairness(b, rel, g)

    opt_dict = {
        'rel': rel.tolist(),
        'g': g.tolist(),
        'max_U': float('-inf'),
        'min_V': float('inf'),
        'max_U_alt': float('inf'),
        'min_V_alt': float('inf'),
        'max_U_perm': None,
        'min_V_perm': None,
        'max_U_alt_perm': None,
        'min_V_alt_perm': None,
    }
    for alpha_val in alpha_list:
        opt_dict[f'opt_tradeoff_alpha_{alpha_val}'] = float('inf')
        opt_dict[f'opt_tradeoff_perm_alpha_{alpha_val}'] = None
    # min_tradeoff, min_tradeoff_perm = float('inf'), None

    # get max util perm
    max_U_perm = np.argsort(-rel, stable=True)
    max_U = f_util(make_permutation_matrix(max_U_perm))
    opt_dict['max_U'] = max_U
    opt_dict['max_U_perm'] = max_U_perm.tolist()

    # get alternate min unfair perm
    # alternate_g = torch.tensor([0, 1]*(int(args.k/2)) + [1]*(args.k%2)).repeat(args.m,1).to(torch.float32)
    g1_idx = np.where(g == 0)[0]
    g2_idx = np.where(g == 1)[0]

    rel_g1 = rel[g1_idx]
    rel_g2 = rel[g2_idx]

    max_U_g1_perm = np.argsort(-rel_g1, stable=True)
    max_U_g2_perm = np.argsort(-rel_g2, stable=True)
    alt_unfair_perm = -1*np.ones_like(max_U_perm)
    alt_unfair_perm[g1_idx] = g1_idx[max_U_g1_perm]
    alt_unfair_perm[g2_idx] = g2_idx[max_U_g2_perm]

    alt_unfair = f_unfair(make_permutation_matrix(alt_unfair_perm))
    opt_dict['max_U_alt'] = alt_unfair
    opt_dict['max_U_alt_perm'] = alt_unfair_perm.tolist()
    # util = f_util(make_permutation_matrix(perm))
    # if util > opt_dict['max_U']:
    #     opt_dict['max_U'] = util.item()
    #     opt_dict['max_U_perm'] = perm.tolist()

    alternate_g = np.array([0, 1]*(int(args.k/2)) + [1]*(args.k%2)).astype(np.float32)
    for _, perm in enumerate(permutations(np.arange(args.k))):
        perm = np.array(perm)
        
        # util = f_util(make_permutation_matrix(perm))
        # if util > opt_dict['max_U']:
        #     opt_dict['max_U'] = util.item()
        #     opt_dict['max_U_perm'] = perm.tolist()

        unfair = f_unfair(make_permutation_matrix(perm))
        if unfair < opt_dict['min_V']:
            opt_dict['min_V'] = unfair
            opt_dict['min_V_perm'] = perm.tolist()

        # g_perm = g[perm]
        # if torch.equal(g_perm, alternate_g):
        #     if unfair < opt_dict['min_V_alt']:
        #         opt_dict['min_V_alt'] = unfair.item()
        #         opt_dict['min_V_alt_perm'] = perm.tolist()
        
        
        for alpha_val in alpha_list:
            f = lambda b: utility_and_unfairness(b, rel, g, alpha=alpha_val)
            tradeoff = f(make_permutation_matrix(perm))
            if tradeoff < opt_dict[f'opt_tradeoff_alpha_{alpha_val}']:
                opt_dict[f'opt_tradeoff_alpha_{alpha_val}'] = tradeoff
                opt_dict[f'opt_tradeoff_perm_alpha_{alpha_val}'] = perm.tolist()

    return opt_dict


def greedy_ranker(args, rel, g):
    f = lambda b: utility_and_unfairness_greedy(b, rel, g, alpha=args.alpha)
    desc_idx = np.argsort(-rel, stable=True)
    desc_group_idx_dict = defaultdict(list)
    # loop to split indices by group
    for d_idx in desc_idx:
        desc_group_idx_dict[g[d_idx]].append(d_idx)
    # initialize the greedy idx
    greedy_idx =  list() #[-1]*idx.shape[0]

    # iterate over position
    for j in range(desc_idx.shape[0]):
        # reset the tradeoff after every position
        tradeoff_list = list()
        seek_indices = list()
        remaining = list()
        # iterate over sorted document lists by group
        for key in desc_group_idx_dict.keys():
            # filter already seen docs
            diff = list(filterfalse(greedy_idx.__contains__,desc_group_idx_dict[key]))
            if diff:
                remaining.append(diff[:args.look_ahead])
            else:
                continue
        remaining = list(chain.from_iterable(remaining))
        # calculate tradeoff for each choice
        for idx in remaining:
            tradeoff_list.append(f(np.array(greedy_idx+[idx])))
            seek_indices.append(greedy_idx+[idx])
        # select the choice with the lowest tradeoff value
        greedy_idx = seek_indices[np.argmin(np.array(tradeoff_list))]

    # accumulate data
    greedy_tradeoff_perm = np.array(greedy_idx)
    greedy_tradeoff = f(greedy_tradeoff_perm)
    greedy_dict = {
        'greedy_tradeoff': greedy_tradeoff,
        'greedy_tradeoff_perm': greedy_tradeoff_perm.tolist()
    }
    return greedy_dict

def get_util_and_unfairness_on_samples(args, rels, gs, alpha_list=[0.5], rank_greedy=False, write_csv_path=None):
    # print(rels, gs)
    results_list = list()
    for i in tqdm(range(len(rels))):
        rel = rels[i]
        g = gs[i]
        results_dict = opt_ranker(args,rel, g, alpha_list)
        if rank_greedy:
            results_dict.update(greedy_ranker(args, rel, g))
        results_list.append(results_dict)
    results_df = pd.DataFrame(results_list)
    if write_csv_path is not None:
        results_df.to_csv(write_csv_path, index=False)
    return results_df
    


def plot_rankings(read_csv_path, write_fig_path=f'./figures/moon_exp.pdf'):
    results_df = pd.read_csv(read_csv_path)
    gs = np.array(results_df['g'].apply(lambda x: eval(x)).tolist())
    rels = np.array(results_df['rel'].apply(lambda x: eval(x)).tolist())
    # print(rels)
    min_V_perm = np.array(results_df['min_V_perm'].apply(lambda x: eval(x)).to_list())
    greedy_perm = np.array(results_df['greedy_tradeoff_perm'].apply(lambda x: eval(x)).to_list())
    max_U_perm = np.array(results_df['max_U_perm'].apply(lambda x: eval(x)).to_list())

    plot_min_V = np.take_along_axis(gs, min_V_perm, axis=1)
    plot_greedy = np.take_along_axis(gs, greedy_perm, axis=1)
    plot_max_U = np.take_along_axis(gs, max_U_perm, axis=1)

    rel_min_V = np.take_along_axis(rels, min_V_perm, axis=1)
    rel_greedy = np.take_along_axis(rels, greedy_perm, axis=1)
    rel_max_U = np.take_along_axis(rels, max_U_perm, axis=1)

    # compute position movement per document (new_pos - old_pos)
    # inverse permutations: for each doc id, its position in each ordering
    pos_in_min = np.argsort(min_V_perm, axis=1, kind='stable')
    pos_in_greedy = np.argsort(greedy_perm, axis=1, kind='stable')
    pos_in_maxU = np.argsort(max_U_perm, axis=1, kind='stable')

    # per-document movement relative to min_V ordering
    move_per_doc_greedy = pos_in_greedy - pos_in_min
    move_per_doc_maxU = pos_in_maxU - pos_in_min

    # gather movement in the display order (columns) of each permutation
    greedy_perm_diff = np.take_along_axis(move_per_doc_greedy, greedy_perm, axis=1)
    max_U_perm_diff = np.take_along_axis(move_per_doc_maxU, max_U_perm, axis=1)

    # annotations
    rel_rank_diff_greedy = np.asarray([f"{val:.2f}\n{label}" for val, label in zip(rel_greedy.flatten(), greedy_perm_diff.flatten())]).reshape(rel_greedy.shape)
    rel_rank_diff_max_U = np.asarray([f"{val:.2f}\n{label}" for val, label in zip(rel_max_U.flatten(), max_U_perm_diff.flatten())]).reshape(rel_max_U.shape)
    
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18,90))
    # print(axes.shape)
    sns.heatmap(plot_min_V, annot=rel_min_V, fmt='.2f', cbar=False, cmap='icefire', linewidths=0.5, xticklabels=False, yticklabels=False, ax=axes[0])
    # print(annot_rank(rel_greedy, greedy_perm_diff))
    sns.heatmap(plot_greedy, annot=rel_rank_diff_greedy, fmt='', cbar=False, cmap='icefire', linewidths=0.5, xticklabels=False, yticklabels=False, ax=axes[1])
    sns.heatmap(plot_max_U, annot=rel_rank_diff_max_U, fmt='', cbar=False, cmap='icefire', linewidths=0.5, xticklabels=False, yticklabels=False, ax=axes[2])
    axes[0].set_title('min V')
    axes[1].set_title(fr'greedy@$\alpha$={args.alpha}')
    axes[2].set_title('max U')
    fig.savefig(write_fig_path, bbox_inches='tight', dpi=300)



def plot_tradeoff_dist(read_csv_path, alpha_list, write_fig_path=f'./figures/moon_exp_objectives.pdf'):
    results_df = pd.read_csv(read_csv_path)
    fig, axes = plt.subplots(ncols=2+len(alpha_list), nrows=1, figsize=(6*(len(alpha_list)+2),6))
    
    sns.histplot(results_df['max_U'], kde=True, ax=axes[0])
    axes[0].set_title('Max U Distribution')
    
    sns.histplot(results_df['min_V'], kde=True, ax=axes[1])
    axes[1].set_title('Min V Distribution')

    for i, alpha_val in enumerate(alpha_list):
        if alpha_val == 0.:
            plot_df = results_df[f'opt_tradeoff_alpha_{alpha_val}']
        else:
            plot_df = results_df[f'opt_tradeoff_alpha_{alpha_val}'].apply(lambda x: -1 * x)
        sns.histplot(plot_df, kde=True, ax=axes[2+i])
        axes[2+i].set_title(fr'Optimal Trade-off Distribution @ $\alpha$={alpha_val}')
    
    fig.savefig(write_fig_path, bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    # sample_random_rels_and_gs(8, 10, 100)
    args = _parse_args(None)
    rng = np.random.default_rng(seed=args.seed)
    # # Exp 1
    # rels, gs = sample_random_rels_and_gs(args)
    # get_util_and_unfairness_on_samples(args, rels, gs, alpha_list=[0.0], write_csv_path=f"{args.exp_path}/moon_exp.csv")
    # plot_rankings(read_csv_path=f'{args.exp_path}/moon_exp.csv')
    # Exp 2
    alpha_list=[0., 0.01, 0.05, 0.1, 0.5, 1.]
    rels, gs = sample_random_rels_and_gs(args, low=500., high=1_000.)
    get_util_and_unfairness_on_samples(args, rels, gs, alpha_list, rank_greedy=False, write_csv_path=f"{args.exp_path}/dominating_U.csv")
    plot_tradeoff_dist(f'{args.exp_path}/dominating_U.csv', alpha_list)
    # Exp 3: plot ranking weight bias
    # Exp 4: see what happens when the proportion of groups is skewed