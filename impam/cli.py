import argparse
import logging
import sys

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.linalg import solve

_logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.DEBUG)


def cmvn(
    Z: np.ndarray,
    Sigma: np.ndarray,
    impute_index: int,
):

    """Conditional multivariate normal"""
    Sigma_uo = np.delete(Sigma[impute_index], impute_index)
    Sigma_o = np.delete(np.delete(Sigma, impute_index, 0), impute_index, 1)
    Z_o = np.delete(Z.values, impute_index)

    EZ_u = Sigma_uo @ solve(Sigma_o, Z_o, assume_a="pos")
    V_Z = Sigma_uo @ (solve(Sigma_o, Sigma_uo, assume_a="pos"))

    return EZ_u, V_Z


def impute_single_Z(
    full_Sigma,
    full_df,
    impute_index,
    impute_window,
    exclude_window,
):
    # get window index
    window_idx = np.abs(full_df["GPOS"] - full_df["GPOS"][impute_index]) < impute_window
    window_idx = np.logical_and(
        window_idx,
        np.abs(full_df["GPOS"] - full_df["GPOS"][impute_index]) > exclude_window,
    )
    window_idx = np.logical_and(window_idx, ~np.isnan(full_df["Z"]))

    window_idx[impute_index] = True

    # get Z and Sigma
    Z_o = full_df[window_idx]["Z"]
    Sigma = full_Sigma[window_idx, :][:, window_idx]
    new_impute_index = np.cumsum(window_idx)[impute_index] - 1

    Z, V = cmvn(Z_o, Sigma, new_impute_index)

    #
    return Z, V


def impute_all_Z(
    full_Sigma,
    full_df,
    impute_indices: np.ndarray = None,
    impute_window=1.0,
    exclude_window=0.0,
):
    impute_Z = np.repeat(np.nan, full_df.shape[0])
    impute_V = np.repeat(np.nan, full_df.shape[0])

    if impute_indices is None:
        impute_indices = np.arange(impute_Z.shape[0])

    for i in impute_indices:
        if i % 100 == 0:
            logging.debug(i)
        impute_Z[i], impute_V[i] = impute_single_Z(
            full_Sigma=full_Sigma,
            full_df=full_df,
            impute_index=i,
            impute_window=impute_window,
            exclude_window=exclude_window,
        )

    return impute_Z, impute_V


def extrapolate_nearest(
    full_df,
    impute_indices: np.ndarray = None,
    impute_window=None,
    exclude_window=0.0,
):

    impute_Z = np.repeat(np.nan, full_df.shape[0])
    if impute_indices is None:
        impute_indices = np.arange(impute_Z.shape[0])

    lookup_table = full_df[["GPOS", "T_STAT"]]

    missing_index = np.isnan(lookup_table["T_STAT"])

    def index_included(x):
        return np.prod(
            [
                ~np.in1d(np.arange(impute_Z.shape[0]), x),
                ~missing_index,
                np.abs(lookup_table["GPOS"] - lookup_table["GPOS"][x]) > exclude_window,
            ],
            axis=0,
        ).astype(bool)

    for i in impute_indices:
        ii = index_included(i)
        if not np.any(ii):
            continue
        extrapolated_Z = scipy.interpolate.interp1d(
            lookup_table["GPOS"][ii],
            lookup_table["T_STAT"][ii],
            kind="nearest",
            fill_value="extrapolate",
        )(full_df["GPOS"][i])

        impute_Z[i] = extrapolated_Z

    return impute_Z


def add_genetic_position(df, gmap):
    gpos = np.interp(
        x=df["POS"].values,
        xp=gmap.iloc[:, 1],
        fp=gmap.iloc[:, 3],
    )
    # df_ = df.copy()
    # df_["GPOS"] = gpos

    return gpos


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

    parser.add_argument(
        "--naive",
        help="impute from the nearest marker",
        action="store_true",
        default=False,
    )
    parser.add_argument("--sumstat", help="Input filename, tab-delimited")
    parser.add_argument("--out", help="Output filename", default=sys.stdout)
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
    parser.add_argument(
        "--markerColName", help="Column name of genetic position", default="ID"
    )
    parser.add_argument(
        "--zColName",
        help="Column name of Z scores in input summary statistics",
        default="Z",
    )
    parser.add_argument(
        "--posColName",
        help="Column name of physical position in input summary statistics",
        default="POS",
    )
    parser.add_argument(
        "--cMColName",
        help="Column name of genetic position in the output",
        default="GPOS",
    )
    parser.add_argument("--cM", help="Imputation window", default=1.0, type=float)
    parser.add_argument("--no-cM", help="Exclusion window", default=0.0, type=float)
    # input targets
    parser.add_argument("--target", help="Position to impute")

    args = parser.parse_args()

    # units

    # Read
    df = pd.read_csv(args.sumstat, sep="\t")

    df = df.rename({args.posColName: "POS"}, axis=1)

    if args.target is not None:
        target_pos = np.loadtxt(args.target)
        # TODO: Add warning, exist pos
        target_pos = target_pos[~np.in1d(target_pos, df["POS"])]
        target_df = df.iloc[
            :0,
        ].copy()
        target_df = target_df.assign(POS=target_pos)
        df = pd.concat([df, target_df], ignore_index=True)

    if args.cMColName not in df.columns and args.gmap is None:
        raise RuntimeError(
            "Please include column <genetic position> or provide genetic map"
        )
    if args.gmap is not None:
        gmap_df = pd.read_csv(args.gmap, sep=" ")
        gpos = add_genetic_position(df, gmap_df)
        df[args.cMColName] = gpos

    logging.warning("Finished preprocessing")
    logging.warning(df.head())

    if not args.naive:
        df_forimpute = df[[args.markerColName, args.cMColName, args.zColName]]
        df_forimpute.columns = ["ID", "GPOS", "Z"]

        # compute correlation
        logging.warning("Computing LAD")
        Sigma = get_Sigma(df_forimpute["GPOS"], g=args.g, epsilon=args.epsilon)

        # impute
        logging.warning("Imputing")
        Z_imputed, V_imputed = impute_all_Z(
            full_Sigma=Sigma,
            full_df=df_forimpute,
            impute_window=args.cM,
            exclude_window=args.no_cM,
        )
        df["Z_IMPUTED"] = Z_imputed
        df["V_IMPUTED"] = V_imputed
        df["OUTLIER_STAT"] = (df_forimpute["Z"] - Z_imputed) / np.sqrt(V_imputed)

    else:
        Z_imputed = extrapolate_nearest(df, exclude_window=args.no_cM)
        df["Z_IMPUTED"] = Z_imputed

    # Write
    df.to_csv(args.out, sep="\t", index=None)


if __name__ == "__main__":
    main(sys.argv[1:])
