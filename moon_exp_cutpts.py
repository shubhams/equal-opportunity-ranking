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
from tqdm import tqdm
from to_the_moon import get_util_and_unfairness_on_samples

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
def sample_random_rels_and_gs(args, low=0.0, high=1.0, g_label=0, r=3):

    # rels = torch.tensor([1.]*args.k).repeat(args.m,1)
    # gs = c.sample(sample_shape=(args.m, args.k)).to(torch.float32) # group memberships
    # if g_label == 0:
    rels = rng.uniform(low=low, high=high, size=(int(args.m), int(r))) # relevances
    gs = np.array([g_label]*(int(r))).repeat(int(args.m),axis=0).reshape(int(args.m), int(r)).astype(np.float32)
    # else:
    # rels = u.sample(sample_shape=(int(args.m), int(args.k - args.r))).reshape(int(args.m), int(args.k - args.r)) # relevances
    # gs = torch.tensor([g_label]*(int(args.k - args.r))).repeat(int(args.m),1).to(torch.float32)
    return rels, gs


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
    fig.savefig(f'./figures/moon_rank_freqs_relA_{relA:.2f}.png', bbox_inches='tight', dpi=300)


def get_most_frequent_g(g_arr, r=3):
    # Count 0s in each column
    column_zero_counts = (g_arr == 0).sum(axis=0)
    # Get indices of columns with the most 0s
    best_indices = np.argsort(column_zero_counts, stable=True)[::-1]
    # Construct the result
    result = np.ones(g_arr.shape[1])
    result[best_indices[:r]] = 0.
    return result


def plot_pattern(args, read_csv_path, relBs):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12,10))
    color_map = plt.get_cmap('tab10').colors 

    for r in range(1, args.k):
    # for r in range(3, 4):
        ratio_list, pattern_min_V_list, pattern_max_U_list = [], [], []
        prev_pattern_min_V, prev_pattern_max_U = None, None
        for relB in relBs:
            print(f"{read_csv_path.format(exp_path=args.exp_path, relB_val=relB, r_val=r)}")
            results_df = pd.read_csv(read_csv_path.format(exp_path=args.exp_path, relB_val=relB, r_val=r))
            gs = np.array(results_df['g'].apply(lambda x: eval(x)).tolist())

            min_V_perm = np.array(results_df['min_V_perm'].apply(lambda x: eval(x)).to_list())
            greedy_perm = np.array(results_df['greedy_tradeoff_perm'].apply(lambda x: eval(x)).to_list())
            max_U_perm = np.array(results_df['max_U_perm'].apply(lambda x: eval(x)).to_list())

            gs_min_V = np.take_along_axis(gs, min_V_perm, axis=1)
            gs_greedy = np.take_along_axis(gs, greedy_perm, axis=1)
            gs_max_U = np.take_along_axis(gs, max_U_perm, axis=1)

            # count frequencies of permutations selected
            # mean
            # g1 = 0.5-(np.nanmean(gs, axis=0))
            # g1_min_V = 0.5-(np.nanmean(gs_min_V, axis=0))
            # g1_max_U = 0.5-(np.nanmean(gs_max_U, axis=0))
            # g1_greedy = 0.5-(np.nanmean(gs_greedy, axis=0))
            # g1 = np.argsort(np.nanmean(gs, axis=0))
            # g1_min_V = np.argsort(np.nanmean(gs_min_V, axis=0))
            # g1_max_U = np.argsort(np.nanmean(gs_max_U, axis=0))
            # g1_greedy = np.argsort(np.nanmean(gs_greedy, axis=0))
            g1 = get_most_frequent_g(gs, r)
            g1_min_V = get_most_frequent_g(gs_min_V, r)
            g1_max_U = get_most_frequent_g(gs_max_U, r)
            g1_greedy = get_most_frequent_g(gs_greedy, r)

            # pattern_min_V = ''.join(['A' if x<r else 'B' for x in g1_min_V])
            # pattern_max_U = ''.join(['A' if x<r else 'B' for x in g1_max_U])
            pattern_min_V = ''.join(['A' if x==0 else 'B' for x in g1_min_V])
            pattern_max_U = ''.join(['A' if x==0 else 'B' for x in g1_max_U])
            print(f"ratio={(1.-relB)/relB:.2f}, ex-post max util:{pattern_max_U}, ex-post max fair:{pattern_min_V}, {g1_min_V=}, mean gs_min_V: {np.nanmean(gs_min_V, axis=0)}")
            if pattern_min_V != prev_pattern_min_V:
                ratio_list.append((1.-relB)/relB)
                pattern_min_V_list.append(pattern_min_V)
                pattern_max_U_list.append(pattern_max_U)
                prev_pattern_min_V = pattern_min_V
                # print(f"ratio={(1.-relB)/relB:.2f}, ex-post max util:{pattern_max_U}, ex-post max fair:{pattern_min_V}, {g1_min_V=}")

        axes.axhline(y=r, color=color_map[(r-1)%len(color_map)], linestyle="--")
        axes.plot(ratio_list, [r for _ in range(len(ratio_list))], 'o-', color=color_map[(r-1)%len(color_map)])
        # axes.annotate(r"$\LaTeX$", (1,0), textcoords="offset points", xytext=(0,10), ha='left', rotation=45)
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

    axes.set_xticks([])

    axes.set_xlabel("rel(A)/rel(B) Ratio")
    axes.set_ylabel("number of A docs")
    # axes.annotate("rel(A)/rel(B)", (0.5, 0.3), xycoords="axes fraction", xytext=(0, 0), textcoords="offset points", ha='center')

    fig.savefig(f'./figures/moon_exp_cutpts_r_{r}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'./figures/moon_exp_cutpts_r_{r}.pdf', bbox_inches='tight', dpi=300)
        

if __name__ == "__main__":
    args = _parse_args(None)
    round_decimals = 2
    np.set_printoptions(precision=round_decimals)
    rng = np.random.default_rng(seed=args.seed)
    

    relBs=np.array([0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.35, 0.30, 0.20])
    # relBs=torch.round(torch.tensor([0.30, 0.20]), decimals=round_decimals)
    if not args.just_plot:
        for i in relBs:
            for r in range(1, args.k):
            # for r in range(3, 4):
                print(f"i: {i:.2f}, ratio={((1.-i)/i):.2f} Group A rels={0.9-i, 1.-i}, Group B rels={i-0.1, i}, r={r}")
                rels_1, gs_1 = sample_random_rels_and_gs(args, low=0.9-i, high=1.-i, g_label=0, r=r)
                rels_2, gs_2 = sample_random_rels_and_gs(args, low=i-0.1, high=i, g_label=1, r=args.k - r)
                # rels_1 = rels_1.unsqueeze(dim=-1)
                # gs_1 = gs_1.unsqueeze(dim=-1)
                # rels_2 = rels_2.unsqueeze(dim=-1)
                # gs_2 = gs_2.unsqueeze(dim=-1)
                print(f"{gs_1[0]=}, {gs_2[0]=}")
                concatenated_rels = np.concatenate([rels_1, rels_2], axis=-1)
                rels = concatenated_rels.reshape(rels_1.shape[0], -1)
                concatenated_gs = np.concatenate([gs_1, gs_2], axis=-1)
                gs = concatenated_gs.reshape(gs_1.shape[0], -1)
                
                get_util_and_unfairness_on_samples(args, rels, gs, alpha_list=[0.0], write_csv_path=f"{args.exp_path}/moon_exp_cutpts_relB_{i:.2f}_r_{r}.csv")
                ## analyze rankings
                # plot_rank_freqs(read_csv_path=f'{args.exp_path}/moon_exp_cutpts_relB_{i:.2f}_r_{r}.csv', relB=i)
        plot_pattern(args, read_csv_path='{exp_path}moon_exp_cutpts_relB_{relB_val:.2f}_r_{r_val}.csv', relBs=relBs)
    else:
        plot_pattern(args, read_csv_path='{exp_path}moon_exp_cutpts_relB_{relB_val:.2f}_r_{r_val}.csv', relBs=relBs)
    
    # print(torch.round(torch.arange(0., 1., 0.1), decimals=round_decimals))
    # print(torch.round(torch.log(torch.arange(1., 11., 1))/torch.log(torch.tensor(10.)), decimals=round_decimals))