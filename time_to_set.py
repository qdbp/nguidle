from math import prod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst


def mk_kills_to_set_dist(base_prob: float, set_levels: List[int]):

    """
    base_prob:
        base probability of any item from the set dropping. It's assume the
        probability of any actual item dropping is base_prob/len(set_levels)
    set_levels:
        the levels of existing items in the set. Those at 100 should be
        included.
    """
    p_item = base_prob / len(set_levels)
    neg_bins = tuple(sst.nbinom(100 - l, p_item) for l in set_levels if l < 100)

    class prod_dist(sst.rv_discrete):
        def _pmf(self, k):
            cdf_k = prod(nb.cdf(k) for nb in neg_bins)
            cdf_k_m1 = prod(nb.cdf(k - 1) for nb in neg_bins)
            pmf = cdf_k - cdf_k_m1
            return pmf

    return prod_dist(name="killstoset")


def print_secs_to_set(
    kill_dist: sst.rv_discrete, ttk: float, boss_chance: float = 0.25
):

    quantiles = [0.5, 0.9, 0.99]
    ppfs = kill_dist.ppf(quantiles)

    for q, ppf in zip(quantiles, ppfs):
        print(
            f"{q * 100:.0f}% chance to get set in "
            f"{ppf * ttk / boss_chance / 3600:.2f} hours"
        )

    return ppfs * ttk / boss_chance / 3600


def main():

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("levels", metavar="LEVEL", nargs="+", type=int)
    parser.add_argument(
        "--base_prob", type=float, help="base probability of loot", default=1.0
    )
    parser.add_argument(
        "--ttk", type=float, default=10, help="time to kill average mob"
    )
    parser.add_argument(
        "--boss_chance",
        type=float,
        default=0.25,
        help="percentage of mobs that are bosses",
    )

    args = parser.parse_args()

    dist = mk_kills_to_set_dist(args.base_prob, args.levels)
    print_secs_to_set(dist, args.ttk, boss_chance=args.boss_chance)


if __name__ == "__main__":
    main()
