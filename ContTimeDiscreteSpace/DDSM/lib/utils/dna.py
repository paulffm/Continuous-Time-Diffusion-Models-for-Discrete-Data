import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import logging
import time
import tqdm
import tabix
import pyBigWig
import pandas as pd
from matplotlib import pyplot as plt
from selene_sdk.targets import Target
import numpy as np

logger = logging.getLogger()


class ModelParameters:
    seifeatures_file = "../data/target.sei.names"
    seimodel_file = "../data/best.sei.model.pth.tar"

    ref_file = "../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ref_file_mmap = "../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap"
    tsses_file = "../data/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv"

    fantom_files = [
        "../data/agg.plus.bw.bedgraph.bw",
        "../data/agg.minus.bw.bedgraph.bw",
    ]
    fantom_blacklist_files = [
        "../data/fantom.blacklist8.plus.bed.gz",
        "../data/fantom.blacklist8.minus.bed.gz",
    ]

    diffusion_weights_file = "steps400.cat4.speed_balance.time4.0.samples100000.pth"

    device = "cuda"
    batch_size = 256
    num_workers = 4

    n_time_steps = 400

    random_order = False
    speed_balanced = True

    ncat = 4

    num_epochs = 200

    lr = 5e-4


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(
        self,
        input_paths,
        features,
        shape,
        blacklists=None,
        blacklists_indices=None,
        replacement_indices=None,
        replacement_scaling_factors=None,
    ):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)]
        )
        self.shape = (len(input_paths), *shape)

    def get_feature_data(
        self, chrom, start, end, nan_as_zero=True, feature_indices=None
    ):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [
                    tabix.open(blacklist) for blacklist in self.blacklists
                ]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(
                        self.blacklists, self.blacklists_indices
                    ):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[
                                blacklist_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0) : int(e) - start] = 0
            else:
                for (
                    blacklist,
                    blacklist_indices,
                    replacement_indices,
                    replacement_scaling_factor,
                ) in zip(
                    self.blacklists,
                    self.blacklists_indices,
                    self.replacement_indices,
                    self.replacement_scaling_factors,
                ):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[
                            blacklist_indices,
                            np.fmax(int(s) - start, 0) : int(e) - start,
                        ] = (
                            wigmat[
                                replacement_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ]
                            * replacement_scaling_factor
                        )

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat
