"""
let's try a set of experiments with 3 A candidates, 3 B candidates, where A relevances are all alpha+noise, and the Bs are all beta+noise
you already tried alpha=beta
and we learned that with your current weight function, the order ABBAAB is clearly preferred to ABABAB -- the utility gap is .47 for ABABAB and it drops to .33 for ABBAAB
now imagine alpha>>beta
if alpha/beta is sufficiently high, then "fairness" will demand the order BBBAAA
proposal: don't call V fairness. Call it a "balance score"
then your goal is to trade off "utility" vs "balance"
question: where are the cut points for alpha/beta that have different optimal orders?
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from to_the_moon import get_util_and_unfairness_on_samples
from utils import sample_groupwise_rels
from collections import Counter

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] =r"\usepackage{xcolor}"
# import matplotlib as mpl
# mpl.rcParams.update({
#     "text.usetex": True,
#     "text.latex.preamble": r"\usepackage{xcolor}"
# })


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6, help='Dimension of vector to sort')
    parser.add_argument('--m', type=int, default=100, help='Number of documents to sample from the bag of documents')
    # parser.add_argument('--r', type=int, default=3, help='Number of documents from group A (0)')
    parser.add_argument('--exp_path', type=str, default="./exp_log/", help='Where to save all stuff')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--beam_size', type=int, default=2, help='Number of candidates to keep after each iteration of beam search')
    parser.add_argument('--look_ahead', type=int, default=1, help='Number of sorted docs to look ahead for greedy/beam search')
    parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off parameter between utility and unfairness')
    parser.add_argument('--just_plot', action=argparse.BooleanOptionalAction, default=False, help='Whether to just plot the results from csv files without running the experiments again')
    return parser.parse_args(args)

# TODO: plot bump chart

def plot_rank_freqs(read_csv_path, relA):
    results_df = pd.read_csv(read_csv_path)
    gs = np.array(results_df['g'].apply(lambda x: eval(x)).tolist())
    # rels = np.array(results_df['rel'].apply(lambda x: eval(x)).tolist())

    min_V_perm = np.array(results_df['min_V_perm'].apply(lambda x: eval(x)).to_list())
    greedy_perm = np.array(results_df['greedy_tradeoff_perm'].apply(lambda x: eval(x)).to_list())
    max_U_perm = np.array(results_df['max_U_perm'].apply(lambda x: eval(x)).to_list())

    gs_min_V = np.take_along_axis(gs, min_V_perm, axis=1)
    gs_greedy = np.take_along_axis(gs, greedy_perm, axis=1)
    gs_max_U = np.take_along_axis(gs, max_U_perm, axis=1)

    # count frequencies of permutations selected
    # mean
    g1 = 0.5-(np.nanmean(gs, axis=0))
    g1_min_V = 0.5-(np.nanmean(gs_min_V, axis=0))
    g1_max_U = 0.5-(np.nanmean(gs_max_U, axis=0))
    g1_greedy = 0.5-(np.nanmean(gs_greedy, axis=0))
    # err
    g1_err = np.std(gs, axis=0)
    g1_min_V_err = np.std(gs_min_V, axis=0)
    g1_max_U_err = np.std(gs_max_U, axis=0)
    g1_greedy_err = np.std(gs_greedy, axis=0)
    
    my_cmap = sns.color_palette("icefire", as_cmap=True)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10,8))
    x = np.arange(len(g1_min_V))+1
    width = 0.1
    # plot data in grouped manner of bar type
    # axes.bar(x-0.2, g1, width=width, label='Uniform Sample', color=my_cmap(0.05))
    axes.bar(x-0.1, g1_min_V, width=width, yerr=g1_min_V_err, label='Ex post max fair', color=my_cmap(0.25))
    axes.bar(x, g1_max_U, width=width, yerr=g1_max_U_err, label='Ex post max utility', color=my_cmap(0.75))
    # axes.bar(x+0.1, g1_greedy, width=width, label='Greedy max fairness', color=my_cmap(0.95))
    axes.set_xlabel("Rank Positions")
    axes.set_ylabel("0.5 - Normalized Frequency of Group A Documents")
    axes.legend()
    axes.set_title("RelA: {:.2f}, RelB: {:.2f}".format(relA, 1-relA))
    fig.savefig(f'./figures/moon_rank_freqs_relA_{relA:.2f}.pdf', bbox_inches='tight', dpi=300)


def get_most_frequent_g(g_arr, r):
    # Count 0s in each column
    column_zero_counts = (g_arr == 0).sum(axis=0)
    # Get indices of columns with the most 0s
    best_indices = np.argsort(column_zero_counts, stable=True)[::-1]
    # Construct the result
    result = np.ones(g_arr.shape[1])
    result[best_indices[:r]] = 0.
    return result


def plot_pattern(args, read_csv_path, relAs, ratios):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12,10))
    color_map = plt.get_cmap('tab10').colors 

    for r in range(1, args.k):
    # for r in range(3, 4):
        ratio_list, pattern_min_V_list, pattern_max_U_list = [], [], []
        prev_pattern_min_V, prev_pattern_max_U = None, None
        for relA, ratio in zip(relAs, ratios):
            print(f"{read_csv_path.format(exp_path=args.exp_path, relA_val=relA, r_val=r)}")
            results_df = pd.read_csv(read_csv_path.format(exp_path=args.exp_path, relA_val=relA, r_val=r))
            gs = np.array(results_df['g'].apply(lambda x: eval(x)).tolist())

            min_V_perm = np.array(results_df['min_V_perm'].apply(lambda x: eval(x)).to_list())
            greedy_perm = np.array(results_df['greedy_tradeoff_perm'].apply(lambda x: eval(x)).to_list())
            max_U_perm = np.array(results_df['max_U_perm'].apply(lambda x: eval(x)).to_list())

            gs_min_V = np.take_along_axis(gs, min_V_perm, axis=1)
            gs_greedy = np.take_along_axis(gs, greedy_perm, axis=1)
            gs_max_U = np.take_along_axis(gs, max_U_perm, axis=1)

            # count frequencies of group A (0) in each position
            g1 = get_most_frequent_g(gs, r)
            g1_min_V = get_most_frequent_g(gs_min_V, r)
            g1_max_U = get_most_frequent_g(gs_max_U, r)
            g1_greedy = get_most_frequent_g(gs_greedy, r)

            pattern_min_V = ''.join(['A' if x==0 else 'B' for x in g1_min_V])
            pattern_max_U = ''.join(['A' if x==0 else 'B' for x in g1_max_U])
            print(f"ratio={ratio}, ex-post max util:{pattern_max_U}, ex-post max fair:{pattern_min_V}, {g1_min_V=}, mean gs_min_V: {np.nanmean(gs_min_V, axis=0)}")
            if pattern_min_V != prev_pattern_min_V:
                ratio_list.append(ratio)
                pattern_min_V_list.append(pattern_min_V)
                pattern_max_U_list.append(pattern_max_U)
                prev_pattern_min_V = pattern_min_V

        axes.axhline(y=r, color=color_map[(r-1)%len(color_map)], linestyle="--")
        axes.plot(ratio_list, [r for _ in range(len(ratio_list))], 'o-', color=color_map[(r-1)%len(color_map)])
        for i, (ratio, pattern_min_V) in enumerate(zip(ratio_list, pattern_min_V_list)):
            # TODO: color the As and Bs in the pattern differently
            axes.annotate(f"{pattern_min_V}", (ratio_list[i], r), textcoords="offset points", xytext=(0,10), ha='left', rotation=45)
            axes.annotate(f"{ratio:.2f}", (ratio_list[i], r-0.15), textcoords="offset points", xytext=(0,-10), ha='left', rotation=315)
    
    
    # turn off border
    # axes.set_axis_off()
    frame1 = plt.gca()
    frame1.set_frame_on(False)
    # Hide the tick labels and tick marks
    frame1.yaxis.set_major_locator(MaxNLocator(integer=True))
    # axes.annotate(r"$\LaTeX$", (1,0), textcoords="offset points", xytext=(0,10), ha='left', rotation=45)

    axes.set_xticks([])

    axes.set_xlabel("rel(A)/rel(B) Ratio")
    axes.set_ylabel("number of A docs")

    fig.savefig(f'./figures/moon_exp_cutpts_{args.k}.pdf', bbox_inches='tight', dpi=300)
        

def count_g_occurrences(g_arr, r):
    # Count 0s in each column
    g1_counts = (g_arr == 0).sum(axis=0)
    g2_counts = (g_arr == 1).sum(axis=0)
    g1_proportions = g1_counts / (g1_counts + g2_counts)
    g2_proportions = g2_counts / (g1_counts + g2_counts)
    return g1_proportions, g2_proportions


def plot_position_freqs(args, read_csv_path, relAs, ratios):
    fig, axes = plt.subplots(ncols=len(relAs), nrows=args.k-1, figsize=(32,15), sharex=True, sharey=True)

    for r in range(1, args.k):
    # for r in range(3, 4):        
        for relA, ratio in zip(relAs, ratios):
            print(f"{read_csv_path.format(exp_path=args.exp_path, relA_val=relA, r_val=r)}")
            results_df = pd.read_csv(read_csv_path.format(exp_path=args.exp_path, relA_val=relA, r_val=r))
            gs = np.array(results_df['g'].apply(lambda x: eval(x)).tolist())

            min_V_perm = np.array(results_df['min_V_perm'].apply(lambda x: eval(x)).to_list())
            greedy_perm = np.array(results_df['greedy_tradeoff_perm'].apply(lambda x: eval(x)).to_list())
            max_U_perm = np.array(results_df['max_U_perm'].apply(lambda x: eval(x)).to_list())

            gs_min_V = np.take_along_axis(gs, min_V_perm, axis=1)
            gs_greedy = np.take_along_axis(gs, greedy_perm, axis=1)
            gs_max_U = np.take_along_axis(gs, max_U_perm, axis=1)

            # count frequencies of group A (0) in each position
            g1_min_V_proportions, g2_min_V_proportions = count_g_occurrences(gs_min_V, r)
            print(f"{g1_min_V_proportions=}, {g2_min_V_proportions=}")
            min_V_pattern = get_most_frequent_g(gs_min_V, r)
            min_V_pattern_txt = ''.join(['A' if x==0 else 'B' for x in min_V_pattern])

            ax_row, ax_col = r-1, relAs.tolist().index(relA)
            axes[ax_row, ax_col].plot(g1_min_V_proportions, label='Group A', color='blue')
            axes[ax_row, ax_col].plot(g2_min_V_proportions, label='Group B', color='orange')
            axes[ax_row, ax_col].set_title(fr"$\alpha$: {ratio:.2f}"f"\n{min_V_pattern_txt}")
    # Set the ticks and ticklabels for all axes
    plt.setp(axes, xticks=np.arange(0, args.k), xticklabels=[f'{i}' for i in range(1, args.k+1)])
    fig.supxlabel('Rank Position')
    fig.supylabel('Proportion in Position')
    handles1, labels1 = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles1, labels1, loc='lower left')
    fig.tight_layout(rect=(0.025,0,1,1))
    fig.savefig(f'./figures/moon_exp_cutpts_position_freqs_{args.k}.pdf', bbox_inches='tight', dpi=300)


def get_pattern(row):
    return ''.join(['A' if x==0 else 'B' for x in row])

def get_pattern_histogram(g_arr):
    patterns = np.apply_along_axis(get_pattern, 1, g_arr).astype(str)
    return Counter(patterns)


def plot_pattern_hist(args, read_csv_path, relAs, ratios):
    fig, axes = plt.subplots(ncols=len(relAs), nrows=args.k-1, figsize=(32,15), sharey=True)

    max_len = -1
    for r in range(1, args.k):
    # for r in range(3, 4):        
        for relA, ratio in zip(relAs, ratios):
            print(f"{read_csv_path.format(exp_path=args.exp_path, relA_val=relA, r_val=r)}")
            results_df = pd.read_csv(read_csv_path.format(exp_path=args.exp_path, relA_val=relA, r_val=r))
            gs = np.array(results_df['g'].apply(lambda x: eval(x)).tolist())

            min_V_perm = np.array(results_df['min_V_perm'].apply(lambda x: eval(x)).to_list())
            greedy_perm = np.array(results_df['greedy_tradeoff_perm'].apply(lambda x: eval(x)).to_list())
            max_U_perm = np.array(results_df['max_U_perm'].apply(lambda x: eval(x)).to_list())

            gs_min_V = np.take_along_axis(gs, min_V_perm, axis=1)
            gs_greedy = np.take_along_axis(gs, greedy_perm, axis=1)
            gs_max_U = np.take_along_axis(gs, max_U_perm, axis=1)

            # count frequencies of group A (0) in each position
            min_V_pattern_counter = get_pattern_histogram(gs_min_V)
            print(f"{min_V_pattern_counter=}")
            min_V_pattern_counter = dict(sorted(min_V_pattern_counter.items(), key=lambda item: item[0]))

            if len(min_V_pattern_counter) > max_len:
                max_len = len(min_V_pattern_counter)
            ax_row, ax_col = r-1, relAs.tolist().index(relA)
            axes[ax_row, ax_col].bar(list(min_V_pattern_counter.keys()), list(min_V_pattern_counter.values()), label='Group A', color='tab:blue', width=0.5)
            plt.setp(axes[ax_row, ax_col].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            axes[ax_row, ax_col].set_title(fr"$\alpha$: {ratio:.2f}")
    # set the x-axis limits for all axes based on the maximum pattern length
    for row in range(len(relAs)):
        for col in range(args.k-1):
            ax = axes[col, row]
            ax.set_xlim(-0.5, max_len - 0.5)

    # Set the ticks and ticklabels for all axes
    fig.supxlabel('Frequency')
    fig.supylabel(r'Pattern for each row $|r|$ = 1,...,5')
    fig.tight_layout(rect=(0.025,0,1,1))
    fig.savefig(f'./figures/moon_exp_pattern_hists_{args.k}.pdf', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    args = _parse_args(None)
    round_decimals = 6
    np.set_printoptions(suppress=True, precision=round_decimals)
    rng = np.random.default_rng(seed=args.seed)
    
    ratios = np.logspace(0.0, 0.3, 15)
    relBs = [1]*len(ratios)  # Keep relBs constant
    relAs = ratios * np.array(relBs)  # Calculate relAs based on the ratios and constant relBs
    if not args.just_plot:
        for relA, relB, ratio in zip(relAs, relBs, ratios):
            for r in range(1, args.k):
            # for r in range(3, 4):
                print(f"relB: {relB}, ratio={ratio} Group A rel={relA}, Group B rel={relB}, r={r}")
                rels_1, gs_1 = sample_groupwise_rels(args, low=relA-0.1, high=relA, g_label=0, r=r)
                rels_2, gs_2 = sample_groupwise_rels(args, low=relB-0.1, high=relB, g_label=1, r=args.k - r)
                print(f"{gs_1[0]=}, {gs_2[0]=}")
                concatenated_rels = np.concatenate([rels_1, rels_2], axis=-1)
                rels = concatenated_rels.reshape(rels_1.shape[0], -1)
                concatenated_gs = np.concatenate([gs_1, gs_2], axis=-1)
                gs = concatenated_gs.reshape(gs_1.shape[0], -1)
                
                get_util_and_unfairness_on_samples(args, rels, gs, alpha_list=[0.0], write_csv_path=f"{args.exp_path}/moon_exp_cutpts_relA_{relA:.2f}_r_{r}.csv")
                ## analyze rankings
                # plot_rank_freqs(read_csv_path=f'{args.exp_path}/moon_exp_cutpts_relB_{relB:.2f}_r_{r}.csv', relB=relB)
        plot_pattern(args, read_csv_path='{exp_path}moon_exp_cutpts_relA_{relA_val:.2f}_r_{r_val}.csv', relAs=relAs, ratios=ratios)
        plot_position_freqs(args, read_csv_path='{exp_path}moon_exp_cutpts_relA_{relA_val:.2f}_r_{r_val}.csv', relAs=relAs, ratios=ratios)
        plot_pattern_hist(args, read_csv_path='{exp_path}moon_exp_cutpts_relA_{relA_val:.2f}_r_{r_val}.csv', relAs=relAs, ratios=ratios)
    else:
        plot_pattern(args, read_csv_path='{exp_path}moon_exp_cutpts_relA_{relA_val:.2f}_r_{r_val}.csv', relAs=relAs, ratios=ratios)
        plot_position_freqs(args, read_csv_path='{exp_path}moon_exp_cutpts_relA_{relA_val:.2f}_r_{r_val}.csv', relAs=relAs, ratios=ratios)
        plot_pattern_hist(args, read_csv_path='{exp_path}moon_exp_cutpts_relA_{relA_val:.2f}_r_{r_val}.csv', relAs=relAs, ratios=ratios)