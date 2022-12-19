"""Visualization function examples for the homework project"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns, networkx


def plot_degree_distribution(G):
    """Plot a degree distribution of a graph

    TODO: log-log binning! To understand this better, check out networksciencebook.com
    """
    plot_df = pd.Series(dict(G.degree)).value_counts().sort_index().to_frame().reset_index()
    plot_df.columns = [
        'k', 'count']
    plot_df['log_k'] = np.log(plot_df['k'])
    plot_df['log_count'] = np.log(plot_df['count'])
    fig, ax = plt.subplots()
    ax.scatter(plot_df['k'], plot_df['count'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.suptitle('Mutual Degree Distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('count_k')


def plot_age_distribution_by_gender(nodes):
    """Plot a histogram where the color represents gender"""
    plot_df = nodes[['AGE', 'gender']].copy(deep=True).astype(float)
    plot_df['gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    sns.histplot(data=plot_df, x='AGE', hue='gender', bins=(np.arange(0, 45, 5) + 15))


def plot_node_degree_by_gender(nodes, G):
    """Plot the average of node degree across age and gender"""
    nodes_w_degree = nodes.set_index('user_id').merge((pd.Series(dict(G.degree)).to_frame()),
                                                      how='left',
                                                      left_index=True,
                                                      right_index=True)
    nodes_w_degree = nodes_w_degree.rename({0: 'degree'}, axis=1)
    plot_df = nodes_w_degree.groupby(['AGE', 'gender']).agg({'degree': 'mean'}).reset_index()
    plot_df['Gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    fig = sns.lineplot(data=plot_df, x='AGE', y='degree', hue='Gender')
    fig.set(xlabel='Age', ylabel='Degree')


def plot_neighbour_connectivity_by_gender(nodes, G):
    """Plot the neighbour connectivity of nodes across age and gender"""
    neighbour_connectivity = nodes.set_index('user_id').merge(
        (pd.Series(networkx.average_neighbor_degree(G)).to_frame()),
        how='left',
        left_index=True,
        right_index=True)
    neighbour_connectivity = neighbour_connectivity.rename({0: 'neighbour_connect'}, axis=1)
    plot_df = neighbour_connectivity.groupby(['AGE', 'gender']).agg({'neighbour_connect': 'mean'}).reset_index()
    plot_df['Gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    fig = sns.lineplot(data=plot_df, x='AGE', y='neighbour_connect', hue='Gender')
    fig.set(xlabel='Age', ylabel='Neighbour connectivity')


def plot_clustering_coefficient_by_gender(nodes, G):
    """Plot the neighbour connectivity of nodes across age and gender"""
    clustering_coefficient = nodes.set_index('user_id').merge((pd.Series(networkx.clustering(G)).to_frame()),
                                                              how='left',
                                                              left_index=True,
                                                              right_index=True)
    clustering_coefficient = clustering_coefficient.rename({0: 'cluster_coeff'}, axis=1)
    plot_df = clustering_coefficient.groupby(['AGE', 'gender']).agg({'cluster_coeff': 'mean'}).reset_index()
    plot_df['Gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    fig = sns.lineplot(data=plot_df, x='AGE', y='cluster_coeff', hue='Gender')
    fig.set(xlabel='Age', ylabel='Clustering coefficient')


def plot_age_relations_heatmap(edges_w_features, gender_pair, normalize='logging'):
    """
    Plot a heatmap that represents the distribution of edges
    Gender pair is an argument of the function:
        0: no filter to gender
        1: F-F connections
        2: M-M connections
        3: M-F connections
    Normalization can be done by rowsum or logging as an argument passed to the function
    """
    plot_df = edges_w_features.groupby(['gender_x', 'gender_y', 'AGE_x', 'AGE_y']).agg({'smaller_id': 'count'})
    if gender_pair == 0:
        plot_df_filtered = plot_df
    if gender_pair == 1:
        plot_df_filtered = plot_df.loc[(0, 0)].reset_index()
    if gender_pair == 2:
        plot_df_filtered = plot_df.loc[(1, 1)].reset_index()
    if gender_pair == 3:
        plot_df_filtered = plot_df.loc[(0, 1)].reset_index()

    plot_df_heatmap = plot_df_filtered.pivot_table(index='AGE_x', columns='AGE_y',
                                                   values='smaller_id').fillna(0)
    if normalize == 'logging':
        plot_df_heatmap_norm = np.log(plot_df_heatmap + 1)
    elif normalize == 'rowsum':
        plot_df_heatmap_norm = plot_df_heatmap.div(plot_df_heatmap.sum(axis=1), axis=0)
    else:
        plot_df_heatmap_norm = plot_df_heatmap

    fig = sns.heatmap(plot_df_heatmap_norm)
    fig.invert_yaxis()

    if normalize == 'logging':
        fig.set(title='Logged number of connections by pair')
    elif normalize == 'rowsum':
        fig.set(title='Proportion of connections by pair to total number of connections in the age group')
    else:
        fig.set(title='Number of connections by pair')

    if gender_pair == 0:
        fig.set(xlabel='Age', ylabel='Age')
    if gender_pair == 1:
        fig.set(xlabel='Age (Female)', ylabel='Age (Female)')
    if gender_pair == 2:
        fig.set(xlabel='Age (Male)', ylabel='Age (Male)')
    if gender_pair == 3:
        fig.set(xlabel='Age (Female)', ylabel='Age (Male)')