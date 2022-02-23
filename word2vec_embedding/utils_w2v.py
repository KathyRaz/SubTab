import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances
import sys
import warnings

sys.path.append('../../')
from associations_rules_summary.utils_code import data_binning
# from general import gen_model_uuid, gen_dir
import os

from timeit import default_timer as timer
from datetime import timedelta

import random

import uuid
import os
# import utils.config
from collections import Counter

BASE_DIR = "models/"


def gen_model_uuid(label=None):
    uu = uuid.uuid4().hex
    # os.path.join(
    # os.mkdir(uu)
    if label:
        return f"{uu}_{label}"
    else:
        return uu


def data_transformation(df, MODEL_UUID, name_dataset, prefix, q=5, precision=0, trim_rows=True, trim_multi_val_col=True,
                        path_df_binned=None,
                        create_mapping=True, path_mapping=None, config_params=None, size_df=100000, nulls_ratio=0.2,
                        portion_of_unique_values=16, save_data=False, print_time=True, binning_method='default'):
    if print_time:
        start_bin = timer()
    dataset_full, mapping_bins_to_values = data_binning(df.copy(), q=q, precision=precision,
                                                        path_df_binned=path_df_binned,
                                                        create_mapping=create_mapping, path_mapping=path_mapping,
                                                        config_params=config_params, nulls_ratio=nulls_ratio,
                                                        binning_method=binning_method, print_process=print_time)

    num_of_rows = dataset_full.shape[0]

    if trim_multi_val_col:
        columns_to_select = list(dataset_full.columns)
        for col in dataset_full.columns:
            num = dataset_full[col].nunique()
            if num > num_of_rows / portion_of_unique_values:  # there are to few representative per value in a column
                columns_to_select.remove(col)
            elif num == 1 and dataset_full[col].isna().any() == False:  # there is only one value in the column
                columns_to_select.remove(col)

    else:
        print("take all")
        columns_to_select = list(df.columns)
    if trim_rows:
        if df.shape[0] > size_df:
            rows_to_select = random.sample(list(df.index), size_df)
        else:
            rows_to_select = df.index

    else:
        rows_to_select = df.index

    df_binned = dataset_full.iloc[rows_to_select, :][columns_to_select]
    df_orig_prune = df.iloc[rows_to_select, :][columns_to_select]
    if save_data:
        dataset_dir = gen_dir(MODEL_UUID, name_dataset + "/datasets", base_dir=prefix)
        binned_dataset_path = os.path.join(dataset_dir, "binned_dataset.csv")
        mapping_path = os.path.join(dataset_dir, "mapping_bins_to_values.csv")
        df_orig_prune_path = os.path.join(dataset_dir, "df_orig_prune.csv")

        df_binned.to_csv(binned_dataset_path)
        mapping_bins_to_values.to_csv(mapping_path)
        df_orig_prune.to_csv(df_orig_prune_path)
    if print_time:
        end_bin = timer()
        print("finished data_binning, it took {}".format(timedelta(seconds=end_bin - start_bin)))

    return df_binned, df_orig_prune, mapping_bins_to_values


def create_initial_emmb_and_summary(config_params, full_df_a, project_id, n_clusters=7):
    name_dataset = 'cyber_{}'.format(project_id)
    MODEL_UUID = None
    MODEL_DESCRIPTION = "{}_binned_data_cell_to_vec".format(name_dataset)
    if not MODEL_UUID:
        MODEL_UUID = gen_model_uuid(MODEL_DESCRIPTION)
        print(MODEL_UUID)
    prefix = ""
    print("started")
    start_total = timer()
    print("data_binning")
    start_bin = timer()
    dataset, full_df, mapping_bins_to_values = data_transformation(MODEL_UUID=MODEL_UUID, name_dataset=name_dataset,
                                                                   prefix=prefix,
                                                                   df=full_df_a.copy(), q=10, precision=0,
                                                                   path_df_binned=None,
                                                                   trim_multi_val_col=True, create_mapping=True,
                                                                   path_mapping=None,
                                                                   config_params=None, save_data=True)
    end_bin = timer()
    print("finished data_binning, it took {}".format(timedelta(seconds=end_bin - start_bin)))
    print("table vectorization")
    start_tab_vec = timer()
    vec_list_w2v_rows, vec_list_w2v_cols, w2v_model, corpus_tuple = create_tab_vec_with_emb(
        prefix=prefix, dataset=dataset, config_params=config_params, CONF=config_params['CONF'],
        MODEL_UUID=MODEL_UUID, name_dataset=name_dataset, save_file=True)
    end_tab_vec = timer()
    print("finished table vectorization, it took {}".format(timedelta(seconds=end_tab_vec - start_tab_vec)))

    print(" selecting_rows_and_columns")
    start_cols_rows = timer()
    summary_w2v_wo_bins = create_summary(full_df=full_df, vec_list_w2v_rows=vec_list_w2v_rows,
                                         vec_list_w2v_cols=vec_list_w2v_cols, clustering_algo='KMeans',
                                         n_clusters=n_clusters)

    end_cols_rows = timer()
    print("for selecting_rows_and_columns, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))

    end_total = timer()
    print("finished creating a summary out of a dataset, it took {}".format(timedelta(seconds=end_total - start_total)))
    return summary_w2v_wo_bins, vec_list_w2v_rows, vec_list_w2v_cols, w2v_model, corpus_tuple, dataset, full_df, mapping_bins_to_values, MODEL_UUID, MODEL_DESCRIPTION, prefix, name_dataset


def gen_dir(model_uuid, added, base_dir='None'):
    #    base_dir=utils.config.BASE_DIR
    if base_dir == 'None':
        base_dir = BASE_DIR

    new_dir = os.path.join(base_dir, model_uuid, added)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    return new_dir


def increment_count(cell_dict, ID, key):
    """
    Puts the key in the dictionary and assigns an ID if does not exist or adds one to repeats if it does.
    Args:
        dict: a dictionary mapping a tuple of strings to a tuple (ID, repeats)
        ID:   an ID to assign if key doesn't exist
        key:  a tuple containing attribute and value
    Return:
        ID:      next ID to assign
        curr_ID: ID of the current key
    """

    if key in cell_dict:
        cell_dict[key] = (cell_dict[key][0], cell_dict[key][1] + 1)
    else:
        cell_dict[key] = (ID, 1)
        ID += 1
    return ID, cell_dict[key][0]


def tokenize_table(table, cell_dict, ID, include_attr=True):
    ttable = np.zeros(table.shape, dtype=np.int64)
    for i, row in enumerate(table.values):
        for (j, cell) in enumerate(row):
            if include_attr:
                key = (table.columns[j], cell)
            else:
                key = cell
            ID, curr_ID = increment_count(cell_dict, ID, key)
            ttable[i, j] = curr_ID
    return ttable, ID


def create_corpus(table, include_attr=False):
    cell_dict = {}
    ID = 0
    tokenized_table = []
    tokenized_tt, ID = tokenize_table(table, cell_dict, ID, include_attr=include_attr)
    # tokenized_table.append(tokenized_tt)
    vocabulary = range(ID)
    reversed_dictionary = dict(zip([x[0] for x in cell_dict.values()], cell_dict.keys()))
    # f = open(output_path, 'wb')
    return tokenized_tt, vocabulary, cell_dict, reversed_dictionary


def textualize_rows(matrix):
    corpus_list = []
    for i in range(matrix.shape[0]):
        corpus_list.append([str(a) for a in matrix[i]])
    return corpus_list


def textualize_rows_and_columns(matrix, col_number=None):
    corpus_list = []
    for i in range(matrix.shape[0]):
        corpus_list.append([str(a) for a in matrix[i]])
    mt = matrix.T
    for i in range(mt.shape[0]):
        if col_number:
            col_vals = np.random.choice(mt[i], size=col_number)  #
        else:
            col_vals = mt[i]
        corpus_list.append([str(a) for a in col_vals])

    return corpus_list


def build_model(corpus_tuple, add_columns=False, vec_size=100, row_window_size=20, col_window_size=20, min_count=2,
                model_type='Word2Vec'):
    tokenized_table, vocabulary, cell_dict, reversed_dictionary = corpus_tuple
    if not add_columns:
        table_corpus = textualize_rows(tokenized_table)
    else:
        table_corpus = textualize_rows_and_columns(tokenized_table, col_number=col_window_size)

    if model_type == 'Word2Vec':
        model = Word2Vec(table_corpus, vector_size=vec_size, window=row_window_size, min_count=min_count, workers=4)
    if model_type == 'Doc2Vec':
        documents = [TaggedDocument(doc, [i]) for i, doc in
                     enumerate(table_corpus)]  # doc2vec: create doc for each row.
        model = Doc2Vec(documents, vector_size=vec_size, window=row_window_size, min_count=min_count, workers=4)

    return model


def get_vector(x, t2vmodel, vec_size=50):
    words = list(t2vmodel.wv.index_to_key)
    return t2vmodel.wv.get_vector(str(x)) if str(x) in words else np.full(vec_size, np.nan)


def calc_tabvec(vec_table, rows_num, cols_num, vec_size, vec_for_rows=False, vec_for_cols=False):
    x = np.array(vec_table).reshape(rows_num, cols_num, vec_size)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        med_cols = np.nanmedian(x, axis=[0])
        med_rows = np.nanmedian(x, axis=[1])
        med_all = np.nanmedian(x, axis=[0, 1])

    dev_rows = np.subtract(x, med_rows[:, np.newaxis, :])
    dev_cols = np.subtract(x, med_cols[np.newaxis, :, :])
    dev_all = np.subtract(x, med_all)

    ns_rows = np.nansum(np.square(dev_rows), axis=1)
    ns_cols = np.nansum(np.square(dev_cols), axis=0)

    if vec_for_rows:
        return ns_rows
    if vec_for_cols:
        return ns_cols

    # create one value for each row (row_vec) or column
    row_vec = ns_rows.sum(axis=0) / ns_rows.shape[0]
    col_vec = ns_cols.sum(axis=0) / ns_cols.shape[0]

    # one vector for the whole table
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        all_vec = np.nanmean(np.square(dev_all), axis=(0, 1))
    return all_vec


def generate_table_vectors(t2vmodel, tokenized_table, vec_size=50, tabvec=True, vec_for_rows=False, vec_for_cols=False):
    vectors = t2vmodel.wv
    vec_table = [get_vector(x, t2vmodel, vec_size=vec_size) for _, x in
                 np.ndenumerate(tokenized_table)]  # np.ndenumerate flattens the matrix and gets the value of each cell.
    # then, we search for it's vector (according to trained model)
    if tabvec:
        if vec_for_rows:
            m = calc_tabvec(vec_table, tokenized_table.shape[0], tokenized_table.shape[1], vec_size,
                            vec_for_rows=vec_for_rows, vec_for_cols=vec_for_cols)
        else:
            m = calc_tabvec(vec_table, tokenized_table.shape[0], tokenized_table.shape[1], vec_size,
                            vec_for_cols=vec_for_cols)

    else:
        m = np.nanmean(vec_table, axis=0)
    return m


def count_hits(col, k):
    topsim = list(col.sort_values()[0:k].index)
    s_label = dataset[topsim[0]]["label"]
    lab_list = [dataset[i]["label"] for i in topsim[1:]]
    return len([l for l in lab_list if l == s_label])


def evaluate_model(dataset, vec_list, k=5):
    similarities = euclidean_distances(vec_list)
    sim = pd.DataFrame(similarities)
    a = sim.apply(lambda x: count_hits(x, k), axis=1)
    a.sum()
    result_score = a.sum() / (len(sim) * (k - 1))
    return result_score


def find_closest_row(c, vec_list):
    closesest_vec = vec_list[0]
    idx = 0
    min_distance = np.abs(euclidean_distances(c.reshape(1, -1), closesest_vec.reshape(1, -1))[0][
                              0])  # reshaping means we have only one sample in each array.
    for i, vec in enumerate(vec_list):
        distance = np.abs(euclidean_distances(c.reshape(1, -1), vec.reshape(1, -1))[0][0])
        if distance < min_distance:
            closesest_vec = vec
            min_distance = distance
            idx = i

    return idx, closesest_vec


def create_tab_vec_with_emb(dataset, config_params, CONF, MODEL_UUID, name_dataset, prefix, save_file=False,
                            print_time=False):
    if print_time:
        print("table vectorization")
        start_tab_vec = timer()
    if config_params['BUILD_CORPUS'] == 'True':
        corpus_dir = gen_dir(MODEL_UUID, name_dataset + "/corpus/", base_dir=prefix)
        corpus_path = os.path.join(corpus_dir, "corpus_tuple.pickle")
        #     if not dataset:
        #         dataset= pickle.load(open(dataset_path,"rb"))

        corpus_tuple = create_corpus(dataset, include_attr=CONF["add_attr"])
        if save_file:
            pickle.dump(corpus_tuple, open(corpus_path, "wb"))
    else:
        corpus_dir = gen_dir(MODEL_UUID, name_dataset + "/", base_dir=prefix)
        corpus_path = os.path.join(corpus_dir, "corpus_tuple.pickle")
        corpus_tuple = pickle.load(open(corpus_path, "rb"))

    # build Word2Vec model

    if config_params['BUILD_MODEL'] == 'True':
        model_dir = gen_dir(MODEL_UUID, name_dataset + "/model", base_dir=prefix)
        model_path = os.path.join(model_dir, "w2v_model.model")
        w2v_model = build_model(corpus_tuple,
                                add_columns=CONF["add_columns"],
                                vec_size=CONF["vector_size"],
                                row_window_size=CONF["row_window_size"],
                                col_window_size=CONF["col_window_size"],
                                min_count=CONF["min_count"])
        if save_file:
            w2v_model.save(model_path)

    else:
        model_dir = gen_dir(MODEL_UUID, name_dataset + "/model", base_dir=prefix)
        model_path = os.path.join(model_dir, "w2v_model.model")
        w2v_model = Word2Vec.load(model_path)

    if config_params['VECTORIZE_TABLES'] == 'True':
        vectors_dir = gen_dir(MODEL_UUID, name_dataset + "/vectors", base_dir=prefix)
        # rows
        vectors_path_rows = os.path.join(vectors_dir, "w2v_rows.npy")
        vec_list_w2v_rows = generate_table_vectors(w2v_model, corpus_tuple[0], vec_for_rows=True, vec_for_cols=False)
        # columns
        vectors_path_cols = os.path.join(vectors_dir, "w2v_cols.npy")
        vec_list_w2v_cols = generate_table_vectors(w2v_model, corpus_tuple[0], vec_for_rows=False, vec_for_cols=True, )
        if save_file:
            with open(vectors_path_rows, 'wb') as f:
                np.save(f, vec_list_w2v_rows)
            with open(vectors_path_cols, 'wb') as f:
                np.save(f, vec_list_w2v_cols)
    else:
        vectors_dir = gen_dir(MODEL_UUID, name_dataset + "/vectors", base_dir=prefix)
        vectors_path_rows = os.path.join(vectors_dir, "w2v_rows.npy")
        with open(vectors_path_rows, 'rb') as f:
            vec_list_w2v_rows = np.load(f)
        # columns
        vectors_path_cols = os.path.join(vectors_dir, "w2v_cols.npy")
        with open(vectors_path_cols, 'rb') as f:
            vec_list_w2v_cols = np.load(f)
    if print_time:
        end_tab_vec = timer()
        print("finished table vectorization, it took {}".format(timedelta(seconds=end_tab_vec - start_tab_vec)))

    return vec_list_w2v_rows, vec_list_w2v_cols, w2v_model, corpus_tuple


def create_summary(full_df, vec_list_w2v_rows, vec_list_w2v_cols=None, clustering_algo='KMeans', n_clusters=10,
                   goal_column=None, print_time=False):
    # TODO continue creating the summary with goal column : remove the goal column from the optional columns,
    # remove the vector of this column from the columns vectors,. then when selecting the columns in the final stage)
    # by their index, select it after removing the goal column. then merge the goal column.
    if print_time:
        print(" selecting_rows_and_columns")
        start_cols_rows = timer()
    if clustering_algo == 'KMeans':
        model_rows = KMeans(n_clusters=n_clusters, random_state=0).fit(vec_list_w2v_rows)
    centroids_rows = model_rows.cluster_centers_

    summary_idx_w2v_rows = []
    for c in centroids_rows:
        idx = np.argmin(np.linalg.norm(c - vec_list_w2v_rows, axis=1))
        summary_idx_w2v_rows.append(idx)

    summary_w2v_wo_bins_all_columns = full_df.iloc[summary_idx_w2v_rows].copy()

    # cols
    if vec_list_w2v_cols is None:
        if print_time:
            end_cols_rows = timer()
            print(
                "for selecting_rows_and_columns, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))
        return summary_w2v_wo_bins_all_columns
    columns_list_for_cluster = list(full_df.columns)
    if goal_column is not None:
        n_clusters = n_clusters-len(goal_column)
        if type(goal_column) == list:
            
            for gc in goal_column:
                columns_list_for_cluster.remove(gc)
                vec_list_w2v_cols = np.delete(vec_list_w2v_cols,list(full_df.columns).index(gc),0)
        else:
            goal_column = [goal_column]
            vec_list_w2v_cols =  np.delete(vec_list_w2v_cols,list(full_df.columns).index(goal_column[0]),0)
        

    if clustering_algo == 'KMeans':
        model_cols = KMeans(n_clusters=n_clusters, random_state=0).fit(vec_list_w2v_cols)
    centroids_cols = model_cols.cluster_centers_
    summary_idx_w2v_cols = []
    for c in centroids_cols:
        idx = np.argmin(np.linalg.norm(c - vec_list_w2v_cols, axis=1))
        summary_idx_w2v_cols.append(idx)
    names_of_columns_for_summary = []
    for cc in summary_idx_w2v_cols:
        names_of_columns_for_summary.append(columns_list_for_cluster[cc])
    if goal_column is not None:
        for gc in goal_column:
            names_of_columns_for_summary.append(gc)
    summary_w2v_wo_bins = summary_w2v_wo_bins_all_columns[names_of_columns_for_summary]
    if print_time:
        end_cols_rows = timer()
        print("for selecting_rows_and_columns, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))
    return summary_w2v_wo_bins


def create_summary_for_filtered_dataset_old(dataset, filtered_df, cell_dict, w2v_model, clustering_algo='Kmeans',
                                            n_clusters=7, print_time=True):
    start_filter_sum = timer()
    explore_1_df_binned = dataset.iloc[filtered_df.index][dataset.columns].copy()
    filtered_df = filtered_df[dataset.columns]
    ttable = np.zeros(filtered_df.shape, dtype=np.int64)

    for i, row in enumerate(explore_1_df_binned.values):
        for (j, cell) in enumerate(row):
            if str((filtered_df.columns[j], cell)[1]) == 'nan':
                for k in cell_dict.keys():
                    if str((filtered_df.columns[j], cell)) == str(k):
                        ttable[i, j] = cell_dict[k][0]
                        break

            else:
                ttable[i, j] = cell_dict[(filtered_df.columns[j], cell)][0]

    vec_list_w2v_rows_filterd = generate_table_vectors(w2v_model, ttable, vec_for_rows=True, vec_for_cols=False)
    vec_list_w2v_cols_filterd = generate_table_vectors(w2v_model, ttable, vec_for_rows=False, vec_for_cols=True)
    end_filter_sum = timer()
    if print_time:
        print("for creating rows and cols vectors, it took {}".format(
            timedelta(seconds=end_filter_sum - start_filter_sum)))
        print(" selecting_rows_and_columns")
    start_cols_rows = timer()
    summary_w2v_wo_bins_filtered = create_summary(filtered_df, vec_list_w2v_rows_filterd,
                                                  vec_list_w2v_cols=vec_list_w2v_cols_filterd,
                                                  clustering_algo=clustering_algo, n_clusters=n_clusters)

    end_cols_rows = timer()
    if print_time:
        print("for selecting_rows_and_columns, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))
    return summary_w2v_wo_bins_filtered


def create_summary_for_filtered_dataset(prefix, MODEL_UUID, name_dataset, filtered_df, clustering_algo='KMeans',
                                        n_clusters=7, print_time=True):
    start_filter_sum = timer()
    model_dir = gen_dir(MODEL_UUID, name_dataset + "/datasets", base_dir=prefix)
    binned_dataset_path = os.path.join(model_dir, "binned_dataset.csv")
    dataset = pd.read_csv(binned_dataset_path).drop('Unnamed: 0', axis=1)

    corpus_dir = gen_dir(MODEL_UUID, name_dataset + "/corpus/", base_dir=prefix)
    corpus_path = os.path.join(corpus_dir, "corpus_tuple.pickle")
    corpus_tuple = pickle.load(open(corpus_path, "rb"))
    cell_dict = corpus_tuple[2]

    model_dir = gen_dir(MODEL_UUID, name_dataset + "/model", base_dir=prefix)
    model_path = os.path.join(model_dir, "w2v_model.model")
    w2v_model = Word2Vec.load(model_path)

    explore_1_df_binned = dataset.iloc[filtered_df.index][dataset.columns].copy()
    filtered_df = filtered_df[dataset.columns]
    ttable = np.zeros(filtered_df.shape, dtype=np.int64)

    for i, row in enumerate(explore_1_df_binned.values):
        for (j, cell) in enumerate(row):
            if str((filtered_df.columns[j], cell)[1]) == 'nan':
                for k in cell_dict.keys():
                    if str((filtered_df.columns[j], cell)) == str(k):
                        ttable[i, j] = cell_dict[k][0]
                        break

            else:
                ttable[i, j] = cell_dict[(filtered_df.columns[j], cell)][0]

    vec_list_w2v_rows_filterd = generate_table_vectors(w2v_model, ttable, vec_for_rows=True, vec_for_cols=False)
    vec_list_w2v_cols_filterd = generate_table_vectors(w2v_model, ttable, vec_for_rows=False, vec_for_cols=True)
    end_filter_sum = timer()
    if print_time:
        print("for creating new table vectors, it took {}".format(timedelta(seconds=end_filter_sum - start_filter_sum)))
    start_cols_rows = timer()
    summary_w2v_wo_bins_filtered = create_summary(filtered_df, vec_list_w2v_rows_filterd,
                                                  vec_list_w2v_cols=vec_list_w2v_cols_filterd,
                                                  clustering_algo=clustering_algo, n_clusters=n_clusters)

    end_cols_rows = timer()
    if print_time:
        print("for summary creation, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))
    return summary_w2v_wo_bins_filtered


def create_summary_for_filtered_dataset_memory(dataset, cell_dict, w2v_model, filtered_df, clustering_algo='KMeans',
                                               take_nulls=False, must_column = [],
                                               n_clusters=7):
    start_filter_sum = timer()
    try:
        explore_1_df_binned = dataset.iloc[filtered_df.index][dataset.columns].copy()
    except:
        explore_1_df_binned = dataset.loc[filtered_df.index][dataset.columns].copy()

    filtered_df = filtered_df[dataset.columns]
    ttable = np.zeros(filtered_df.shape, dtype=np.int64)
    if take_nulls:
        for i, row in enumerate(explore_1_df_binned.values):
            for (j, cell) in enumerate(row):
                if str((filtered_df.columns[j], cell)[1]) == 'nan':
                    for k in cell_dict.keys():
                        if str((filtered_df.columns[j], cell)) == str(k):
                            ttable[i, j] = cell_dict[k][0]
                            break

                else:
                    ttable[i, j] = cell_dict[(filtered_df.columns[j], cell)][0]
    else:
        for i, row in enumerate(explore_1_df_binned.values):
            for (j, cell) in enumerate(row):
                ttable[i, j] = cell_dict[(filtered_df.columns[j], cell)][0]
    vec_list_w2v_rows_filtered = generate_table_vectors(w2v_model, ttable, vec_for_rows=True, vec_for_cols=False)
    vec_list_w2v_cols_filtered = generate_table_vectors(w2v_model, ttable, vec_for_rows=False, vec_for_cols=True)
    end_filter_sum = timer()
    print("for creating new table vectors, it took {}".format(timedelta(seconds=end_filter_sum - start_filter_sum)))
    start_cols_rows = timer()
    if len(must_column)>0:
        summary_w2v_wo_bins_filtered = create_summary(filtered_df, vec_list_w2v_rows_filtered,
                                                  vec_list_w2v_cols=vec_list_w2v_cols_filtered,
                                                  clustering_algo=clustering_algo, n_clusters=n_clusters, goal_column=must_column)
    else:
        summary_w2v_wo_bins_filtered = create_summary(filtered_df, vec_list_w2v_rows_filtered,
                                                  vec_list_w2v_cols=vec_list_w2v_cols_filtered,
                                                  clustering_algo=clustering_algo, n_clusters=n_clusters)

    end_cols_rows = timer()
    print("for summary creation, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))
    return summary_w2v_wo_bins_filtered
