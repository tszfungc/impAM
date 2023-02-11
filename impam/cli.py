import argparse
import logging
import sys

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(message)s")


def cmvn(
    Z: np.ndarray,
    Sigma: np.ndarray,
    impute_index: int,
):

    """Conditional multivariate normal"""
    Sigma_uo = np.delete(Sigma[impute_index], impute_index)
    Sigma_o = np.delete(np.delete(Sigma, impute_index, 0), impute_index, 1)
    Z_o = np.delete(Z.values, impute_index)

    EZ_u = Sigma_uo @ np.linalg.solve(Sigma_o, Z_o)
    # TODO: Remove unnecessary ecode
    ecode = 0

    return EZ_u, ecode


def impute_single_Z(full_Sigma, full_df, impute_index, impute_window):
    # get window index
    window_idx = np.abs(full_df["GPOS"] - full_df["GPOS"][impute_index]) < impute_window

    # get Z and Sigma
    Z_o = full_df[window_idx]["Z"]
    Sigma = full_Sigma[window_idx, :][:, window_idx]
    new_impute_index = np.cumsum(window_idx)[impute_index] - 1

    Z = cmvn(Z_o, Sigma, new_impute_index)

    #
    return Z


def impute_all_Z(
    full_Sigma,
    full_df,
    impute_indices: np.ndarray = None,
    impute_window=1.0,
):
    impute_Z = np.repeat(np.nan, full_df.shape[0])
    ecodes = np.zeros(full_df.shape[0], dtype=int)

    if impute_indices is None:
        impute_indices = np.arange(impute_Z.shape[0])

    for i in impute_indices:
        impute_Z[i], ecodes[i] = impute_single_Z(full_Sigma, full_df, i, impute_window)

    return impute_Z, ecodes


def add_genetic_position(df, gmap):
    gpos = np.interp(
        x=df["POS"].values,
        xp=gmap.iloc[:, 1],
        fp=gmap.iloc[:, 3],
    )
    df_ = df.copy()
    df_["GPOS"] = gpos

    return df_


def get_Sigma(
    genetic_distance: np.ndarray,
    g: float,
    epsilon: float,
):
    dist_mat = np.abs(genetic_distance.values - genetic_distance.values.reshape(-1, 1))
    return np.exp(-0.01 * dist_mat * g) + np.eye(dist_mat.shape[0]) * epsilon


def main(argv):
    _logger.warning("Imputation of Z-scores in admixture mapping studies")

    # Parser
    parser = argparse.ArgumentParser(
        description="Imputation of Z-scores in admixture mapping studies"
    )

    parser.add_argument("--sumstat", help="Input filename, tab-delimited")
    parser.add_argument(
        "--g", help="Number of generations since admixture", type=float, default=12.0
    )
    parser.add_argument("--gmap", help="Genetic map")
    parser.add_argument("--chr", help="Chromosomes to be read in genetic map")
    parser.add_argument(
        "--epsilon",
        help="Small variance added to local ancestry correlation",
        type=float,
        default=1e-3,
    )
    # genetic position col or genetic map+physical position
    parser.add_argument("--out", help="Output filename", default=sys.stdout)
    parser.add_argument("--Zcolname", help="Column name of Z scores", default="Z")

    args = parser.parse_args()

    # units

    # Read
    df = pd.read_csv(args.sumstat, sep="\t")
    if "GPOS" not in df.columns and args.gmap is None:
        raise RuntimeError(
            "Please include column <genetic position> or provide genetic map"
        )
    if args.gmap is not None:
        gmap_df = pd.read_csv(args.gmap, sep=" ")
        df = add_genetic_position(df, gmap_df)

    df_forimpute = df[["ID", "GPOS", args.Zcolname]]
    df_forimpute.columns = ["ID", "GPOS", "Z"]

    # compute correlation
    Sigma = get_Sigma(df_forimpute["GPOS"], g=args.g, epsilon=args.epsilon)

    # impute
    Z_imputed, ecodes = impute_all_Z(Sigma, df_forimpute)

    df["Z_IMPUTED"] = Z_imputed
    df["IMPUTE_ERR"] = ecodes

    # Write
    df.to_csv(args.out, sep="\t")


if __name__ == "__main__":
    main(sys.argv[1:])
