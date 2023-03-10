"""Visualization function examples for the homework project"""
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
import seaborn as sns


def plot_degree_distribution(G):
    """Plot a degree distribution of a graph"""
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


def data_node_degree_by_gender(nodes, G):
    """Return the data needed for the plot of the average of node degree across age and gender"""
    nodes_w_degree = nodes.set_index('user_id').merge((pd.Series(dict(G.degree)).to_frame()),
                                                      how='left',
                                                      left_index=True,
                                                      right_index=True)
    nodes_w_degree = nodes_w_degree.rename({0: 'degree'}, axis=1)
    plot_df = nodes_w_degree.groupby(['AGE', 'gender']).agg({'degree': 'mean'}).reset_index()
    plot_df['Gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    return plot_df


def data_neighbour_connectivity_by_gender(nodes, G):
    """Return the data needed for the plot of the neighbour connectivity of nodes across age and gender"""
    neighbour_connectivity = nodes.set_index('user_id').merge(
        (pd.Series(networkx.average_neighbor_degree(G)).to_frame()),
        how='left',
        left_index=True,
        right_index=True)
    neighbour_connectivity = neighbour_connectivity.rename({0: 'neighbour_connect'}, axis=1)
    plot_df = neighbour_connectivity.groupby(['AGE', 'gender']).agg({'neighbour_connect': 'mean'}).reset_index()
    plot_df['Gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    return plot_df


def data_clustering_coefficient_by_gender(nodes, G):
    """Return the data needed for the plot of the neighbour connectivity of nodes across age and gender"""
    clustering_coefficient = nodes.set_index('user_id').merge((pd.Series(networkx.clustering(G)).to_frame()),
                                                              how='left',
                                                              left_index=True,
                                                              right_index=True)
    clustering_coefficient = clustering_coefficient.rename({0: 'cluster_coeff'}, axis=1)
    plot_df = clustering_coefficient.groupby(['AGE', 'gender']).agg({'cluster_coeff': 'mean'}).reset_index()
    plot_df['Gender'] = plot_df['gender'].replace({0.0: 'woman', 1.0: 'man'})
    return plot_df


def plot_figure_three(nodes, G):
    """
    Creating a plot with reference to Figure 3 of the example article
    The subplots in this figure include the data of the previous functions
        a) Degree centrality by gender
        b) Neighbour connectivity by gender
        c) Clustering coefficients by gender
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    sns.lineplot(data=data_node_degree_by_gender(nodes, G), x='AGE', y='degree', hue='Gender',
                 ax=ax1)
    ax1.set(xlabel='Age', ylabel='Degree', title='(a) Degree centrality')
    sns.lineplot(data=data_neighbour_connectivity_by_gender(nodes, G), x='AGE', y='neighbour_connect', hue='Gender',
                 ax=ax2)
    ax2.set(xlabel='Age', ylabel='Neighbour connectivity', title='(b) Neighbour connectivity')
    sns.lineplot(data=data_clustering_coefficient_by_gender(nodes, G), x='AGE', y='cluster_coeff', hue='Gender',
                 ax=ax3)
    ax3.set(xlabel='Age', ylabel='Clustering coefficient', title='(c) Clustering coefficient')
    plt.tight_layout()


def data_age_relations_heatmap(edges_w_features, gender_pair, normalize='logging'):
    """
    Return the data needed for the plot a heatmap that represents the distribution of edges
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
    return plot_df_heatmap_norm


def plot_figure_five(edges_w_features, normalize='logging'):
    """
        Creating a plot with reference to Figure 5 of the example article
        The subplots in this figure include the data of the previous functions
            a) Number of connections per pair
            b) Number of Female-Female connections per pair
            c) Number of Male-Male connections per pair
            d) Number of Male-Female connections per pair
    """
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(14, 10))
    sns.heatmap(data_age_relations_heatmap(edges_w_features=edges_w_features, gender_pair=0, normalize=normalize),
                ax=ax1)
    ax1.invert_yaxis()
    ax1.set(xlabel="Age", ylabel="Age", title="(a) #connections per pair")
    sns.heatmap(data_age_relations_heatmap(edges_w_features=edges_w_features, gender_pair=1, normalize=normalize),
                ax=ax2)
    ax2.invert_yaxis()
    ax2.set(xlabel="Age (Female)", ylabel="Age (Female)", title="(b) #connections per F-F pair")
    sns.heatmap(data_age_relations_heatmap(edges_w_features=edges_w_features, gender_pair=2, normalize=normalize),
                ax=ax3)
    ax3.invert_yaxis()
    ax3.set(xlabel="Age (Male)", ylabel="Age (Male)", title="(c) #connections per M-M pair")
    sns.heatmap(data_age_relations_heatmap(edges_w_features=edges_w_features, gender_pair=3, normalize=normalize),
                ax=ax4)
    ax4.invert_yaxis()
    ax4.set(xlabel="Age (Male)", ylabel="Age (Female)", title="(d) #connections per M-F pair")

    if normalize == "rowsum":
        plt.suptitle('All values are proportionate to the sum of total connections in a given age group for X-axis')
    if normalize == "logging":
        plt.suptitle('All values are log-scaled')
    plt.tight_layout()


def pivot_proportions(edges_w_features, base_gender, out_gender):
    """
    The function calculates the proportion of connections to a given gender to all connections for a given gender
    If base_gender = 0 and out_gender = 1, then the function calculates the proportion of male connections to
    the total connections for female users.
    :param edges_w_features: edges with features
    :param base_gender: 0 - Male, 1 - Female; for which X-axis we would like to plot
    :param out_gender: 0 - Male, 1 - Female; for which proportion of connections are we looking for
    :return: pivot table
    """

    if base_gender == 0 and out_gender == 0:
        df_pivot = data_age_relations_heatmap(edges_w_features, gender_pair=1, normalize="other")
        df_pivot["T_total"] = df_pivot.sum(axis=1) + data_age_relations_heatmap(edges_w_features, 3,
                                                                                normalize="other").sum(axis=1)
    if base_gender == 1 and out_gender == 1:
        df_pivot = data_age_relations_heatmap(edges_w_features, gender_pair=2, normalize="other")
        df_pivot["T_total"] = df_pivot.sum(axis=1) + data_age_relations_heatmap(edges_w_features, 3,
                                                                                normalize="other").transpose().sum(
            axis=1)
    if base_gender == 1 and out_gender == 0:
        df_pivot = data_age_relations_heatmap(edges_w_features, gender_pair=3, normalize="other").transpose()
        df_pivot["T_total"] = df_pivot.sum(axis=1) + data_age_relations_heatmap(edges_w_features, 2,
                                                                                 normalize="other").sum(
            axis=1)
    if base_gender == 0 and out_gender == 1:
        df_pivot = data_age_relations_heatmap(edges_w_features, gender_pair=3, normalize="other")
        df_pivot["T_total"] = df_pivot.sum(axis=1) + data_age_relations_heatmap(edges_w_features, 1,
                                                                                normalize="other").sum(axis=1)

    df_proportion = df_pivot
    df_proportion.iloc[:, :-1] = df_pivot.iloc[:, :-1].div(df_pivot["T_total"], axis=0)
    return df_proportion


def data_gen_proportion(df_proportion):
    """
    This function calculates the proportion of age groups within a given gender.
    The calculation is based on the output of the previous function, which calculates the proportion of all ages to
    the total. This function sums the ages together for a given group based on the base node's age.
    Younger generation means 10 to 20 years younger, than base node
    Same generation means within 5-year-old range with the base node
    Older generation means 10 to 10 years older, than base node
    :param df_proportion: a dataframe with the output of the previous pivot_proportions function
    :return: returns dataframe with three additional columns for percentages of each three age groups
    """

    # creating generations, but because of the smaller sample (ages between 15-50 we declared different ranges
    # (same = +- 5 years, older = +10 to +20, younger = -10 to -20 years old)
    same_gen = {}
    older_gen = {}
    younger_gen = {}

    for age in df_proportion.index.astype(int):
        same_gen[age] = sum(
            df_proportion.loc[age, range(max(age - 5, 15), min(age + 5, 50))]
        )

    for age in df_proportion.index.astype(int):
        if age <= 40:
            older_gen[age] = sum(
                df_proportion.loc[age, range(age + 10, min(age + 20, 50))]
            )
        # because data is out of the sample
        else:
            older_gen[age] = 0

    for age in df_proportion.index.astype(int):
        if age >= 25:
            younger_gen[age] = sum(
                df_proportion.loc[age, range(max(age - 20, 15), max(age - 10, 15))]
            )
        # because data is out of the sample
        else:
            younger_gen[age] = 0

    plot_data_df_proportion = df_proportion
    plot_data_df_proportion["(x-5:x+5)"] = same_gen.values()
    plot_data_df_proportion["(x+10:x+20)"] = older_gen.values()
    plot_data_df_proportion["(x-20:x-10)"] = younger_gen.values()
    return plot_data_df_proportion


def plot_figure_six(edges_w_features):
    """
    This function creates Figure 6, by using previous functions to
    calculate the proportion of friends by gender and age group for both genders.
    """

    # Creating input for the plot, with all 4 combination of base gender and analysed gender

    input_1 = pivot_proportions(edges_w_features, base_gender=0, out_gender=0)
    input_2 = pivot_proportions(edges_w_features, base_gender=0, out_gender=1)
    input_3 = pivot_proportions(edges_w_features, base_gender=1, out_gender=1)
    input_4 = pivot_proportions(edges_w_features, base_gender=1, out_gender=0)

    # Calculating the proportions for each age group by gender

    plot_input_1 = data_gen_proportion(input_1)
    plot_input_2 = data_gen_proportion(input_2)
    plot_input_3 = data_gen_proportion(input_3)
    plot_input_4 = data_gen_proportion(input_4)

    # Creating two charts with the 4 inputs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    sns.lineplot(data=plot_input_1, x="AGE_x", y="(x-5:x+5)", color="red", ax=ax1)
    sns.lineplot(
        data=plot_input_1, x="AGE_x", y="(x+10:x+20)", color="green", ax=ax1
    )
    sns.lineplot(data=plot_input_1, x="AGE_x", y="(x-20:x-10)", color="cyan", ax=ax1)
    sns.lineplot(data=plot_input_2, x="AGE_x", y="(x-5:x+5)", color="blue", ax=ax1)
    sns.lineplot(
        data=plot_input_2, x="AGE_x", y="(x+10:x+20)", color="deeppink", ax=ax1
    )
    sns.lineplot(
        data=plot_input_2, x="AGE_x", y="(x-20:x-10)", color="black", ax=ax1
    )
    ax1.set(ylabel="Proportions", xlabel="Age x of Female User", title="(a) Proportion of Female???s friends??? age")
    ax1.legend(
        [
            "F(x-5:x+5)",
            "F(x+10:x+20)",
            "F(x-20:x-10)",
            "M(x-5:x+5)",
            "M(x+10:x+20)",
            "M(x-20:x-10)",
        ]
    )

    sns.lineplot(data=plot_input_3, x="AGE_x", y="(x-5:x+5)", color="red", ax=ax2)
    sns.lineplot(
        data=plot_input_3, x="AGE_x", y="(x+10:x+20)", color="green", ax=ax2
    )
    sns.lineplot(data=plot_input_3, x="AGE_x", y="(x-20:x-10)", color="cyan", ax=ax2)
    sns.lineplot(data=plot_input_4, x="AGE_y", y="(x-5:x+5)", color="blue", ax=ax2)
    sns.lineplot(
        data=plot_input_4, x="AGE_y", y="(x+10:x+20)", color="deeppink", ax=ax2
    )
    sns.lineplot(
        data=plot_input_4, x="AGE_y", y="(x-20:x-10)", color="black", ax=ax2
    )
    ax2.set(ylabel="Proportions", xlabel="Age x of Male User", title="(b) Proportion of Male???s friends??? age")
    ax2.legend(
        [
            "M(x-5:x+5)",
            "M(x+10:x+20)",
            "M(x-20:x-10)",
            "F(x-5:x+5)",
            "F(x+10:x+20)",
            "F(x-20:x-10)",
        ]
    )
