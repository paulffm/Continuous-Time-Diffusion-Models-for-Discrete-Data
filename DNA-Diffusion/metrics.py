from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

def motif_scoring_KL_divergence(original: pd.Series, generated: pd.Series) -> torch.Tensor:
    """
    This function encapsulates the logic of evaluating the KL divergence metric
    between two sequences.
    Returns
    -------
    kl_divergence: Float
    The KL divergence between the input and output (generated)
    sequences' distribution
    """
    #When calculating KL we are measuring occurances of particular motifs. That means we are not 
    # measuring distribution between "letters". This was the assumption from the original nb. We 
    # are also not measuring occurances at particular place, just wether and how much a motif occures 
    # between the train and test.

    kl_pq = kl_div(original, generated)
    return np.sum(kl_pq)

def motif_scoring_JS_divergence(original: pd.Series, generated: pd.Series) -> torch.Tensor:
    """
    This function encapsulates the logic of evaluating the Jensen-Shannon divergence metric
    between two sequences.
    Returns
    -------
    JS_divergence: Float
    The JS divergence between the input and output (generated)
    sequences' distribution
    """

    js_pq = jensenshannon(original, generated)
    return np.sum(js_pq)

def compare_motif_list(
    df_motifs_a, df_motifs_b, motif_scoring_metric=motif_scoring_KL_divergence, plot_motif_probs=True
):
    """
    This function encapsulates the logic of evaluating the difference between the distribution
    of frequencies between generated (diffusion/df_motifs_a) and the input (training/df_motifs_b) for an arbitrary metric ("motif_scoring_metric")

    Please note that some metrics, like KL_divergence, are not metrics in official sense. Reason
    for that is that they dont satisfy certain properties, such as in KL case, the simmetry property.
    Hence it makes a big difference what are the positions of input.
    """
    set_all_mot = set(df_motifs_a.index.values.tolist() + df_motifs_b.index.values.tolist())
    create_new_matrix = []
    for x in set_all_mot:
        list_in = []
        list_in.append(x)  # adding the name
        if x in df_motifs_a.index:
            list_in.append(df_motifs_a.loc[x][0])
        else:
            list_in.append(1)

        if x in df_motifs_b.index:
            list_in.append(df_motifs_b.loc[x][0])
        else:
            list_in.append(1)

        create_new_matrix.append(list_in)

    df_motifs = pd.DataFrame(create_new_matrix, columns=['motif', 'motif_a', 'motif_b'])

    df_motifs['Diffusion_seqs'] = df_motifs['motif_a'] / df_motifs['motif_a'].sum()
    df_motifs['Training_seqs'] = df_motifs['motif_b'] / df_motifs['motif_b'].sum()
    if plot_motif_probs:
        plt.rcParams["figure.figsize"] = (3, 3)
        sns.regplot(x='Diffusion_seqs', y='Training_seqs', data=df_motifs)
        plt.xlabel('Diffusion Seqs')
        plt.ylabel('Training Seqs')
        plt.title('Motifs Probs')
        plt.show()

    return motif_scoring_metric(df_motifs['Diffusion_seqs'].values, df_motifs['Training_seqs'].values)

def metric_comparison_between_components(original_data, generated_data, cell_components, x_label_plot, y_label_plot):
    """
    This functions takes as inputs dictionaries, which contain as keys different components (cell types)
    and as values the distribution of occurances of different motifs. These two dictionaries represent two different datasets, i.e.
    generated dataset and the input (train) dataset.

    The goal is to then plot a the main evaluation metric (KL or otherwise) across all different types of cell types
    in a heatmap fashion.
    """
    final_comparison_all_components = []
    for components_1, motif_occurance_frequency in original_data.items():
        comparisons_single_component = []
        for components_2 in generated_data.keys():
            compared_motifs_occurances = compare_motif_list(motif_occurance_frequency, generated_data[components_2])
            comparisons_single_component.append(compared_motifs_occurances)

        final_comparison_all_components.append(comparisons_single_component)

    plt.rcParams["figure.figsize"] = (10, 10)
    df_plot = pd.DataFrame(final_comparison_all_components)

    ### caution constant missing
    CELL_NAMES = "CONSTANT"
    ### caution

    df_plot.columns = [CELL_NAMES[x] for x in cell_components]
    df_plot.index = df_plot.columns
    sns.heatmap(df_plot, cmap='Blues_r', annot=True, lw=0.1, vmax=1, vmin=0)
    plt.title(f'Kl divergence \n {x_label_plot} sequences x  {y_label_plot} sequences \n MOTIFS probabilities')
    plt.xlabel(f'{x_label_plot} Sequences  \n(motifs dist)')
    plt.ylabel(f'{y_label_plot} \n (motifs dist)')