from typing import Any

import pandas as pd
import itertools
from random import sample
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import json
import itertools
from tqdm import tqdm
import sys, os
from collections import defaultdict
from efficient_apriori import apriori
from IPython.display import display

filename = sys.argv[0]
os.path.abspath(filename + "/..")



def count_distinct_values(dataset):
    sum_unique = 0
    for col in dataset.columns:
        sum_unique+=dataset[col].nunique()
    return sum_unique 




def save_to_csv(df, path):
    try:
        df.to_csv(path)
    except:
        print("The path {} doesn't exist".format(path))



def create_config_params(df):
    cols_transoform = []
    config_params = {}
    for col in df.columns:
        if df[col].nunique() > 15:
            if df[col].dtype == 'O':  # probably numeric column
                pass
            else:
                cols_transoform.append(col)
    config_params['cols_to_bin'] = cols_transoform

    return config_params


def data_binning(df, q=5, precision=0, path_df_binned=None, create_mapping=True, path_mapping=None, config_params=None,
                 nulls_ratio=0.2, binning_method='default', print_process=False):
    mapping = []
    count_null_cols = 0
    if config_params is None:
        if print_process:
            print("created config params")
        config_params = create_config_params(df)
    num_rows = df.shape[0]
    for col in config_params['cols_to_bin']:  
        if binning_method == 'default':
            if print_process:
                print("default binning")
            if df[col].nunique() > num_rows * 0.8:
                q = 8
            elif df[col].nunique() > num_rows * 0.6:
                q = 6
            elif df[col].nunique() > num_rows * 0.4:
                q = 4
        else:
            if print_process:
                print("{} binning method".format(binning_method))
            q = q

        df_binned = pd.qcut(df[col], q=q, precision=precision, duplicates='drop')
        num_of_labels = df_binned.nunique()
        df[col] = pd.qcut(df[col], q=q, labels=[str(col) + "_" + str(x) for x in range(0, num_of_labels)],
                          precision=precision, duplicates='drop')
        include_nulls = False
        if df[df[col].isna()].shape[0] / df.shape[0] < nulls_ratio and df[df[col].isna()].shape[0] >0:
            if print_process:
                print("column {} has null values that would be taken into consideration".format(col))
                count_null_cols +=1
                print("transforming column ",col)
            df[col] = df[col].astype('str').replace('nan', str(col) + "_null")
            include_nulls = True
            if print_process:
                print("now columns {} has those unique values:".format(col))
                print(df[col].unique())

        if create_mapping:
            values = df_binned.unique().sort_values()
            labels = [str(col) + "_" + str(x) for x in range(0, df_binned.nunique())]
            mapping.append([[labels[i], col, values[i]] for i in range(0, len(labels))])
            if include_nulls:
                if col + '_null' in df[col]:
                    mapping.append([[col + '_null', col, np.NaN]])

    if path_df_binned:
        save_to_csv(df, path_df_binned)

    if create_mapping:
        mapping_flat = list(itertools.chain(*mapping))
        mapping_bins_to_values = pd.DataFrame(mapping_flat, columns=['bin', 'orig_column', 'value'])
        if path_mapping:
            save_to_csv(mapping_bins_to_values, path_mapping)
        return df, mapping_bins_to_values
    if print_process:
        print("we transformed {} cols that had null values".format(count_null_cols))
    return df



def comp_notnull1(df1):
    return [{k: v for k, v in m.items() if pd.notnull(v)} for m in df1.to_dict(orient='records')]


def add_obj_col_manually(rules_df_pos_filtered, rules_df_neg_filtered):
    rules_df_pos_filtered.loc[:, 'rhs'] = rules_df_pos_filtered.loc[:, 'rhs'].str.replace(r'\)$', '', regex=True)
    rules_df_neg_filtered.loc[:, 'rhs'] = rules_df_neg_filtered.loc[:, 'rhs'].str.replace(r'\)$', '', regex=True)
    rules_df_pos_filtered.loc[:, 'rhs'] = rules_df_pos_filtered.loc[:, 'rhs'].apply(
        lambda x: x + str("('CANCELLED,1'))") if x[-1] == ',' else x + str(",('CANCELLED,1'))"))
    rules_df_neg_filtered.loc[:, 'rhs'] = rules_df_neg_filtered.loc[:, 'rhs'].apply(
        lambda x: x + str("('CANCELLED,0'))") if x[-1] == ',' else x + str(",('CANCELLED,0'))"))
    rules_df_pos_filtered.loc[:, 'len_r'] = rules_df_pos_filtered.loc[:, 'len_r'] + 1
    rules_df_neg_filtered.loc[:, 'len_r'] = rules_df_neg_filtered.loc[:, 'len_r'] + 1
    return rules_df_pos_filtered, rules_df_neg_filtered


# filter the rules with given min_sup
def concat_and_filter_rules(rules_dict,
                            name_idx_rule='rule_id',
                             path=None):
    all_rules_df = pd.DataFrame()
    for bin_val in rules_dict.keys():
        rules_df = rules_dict[bin_val]
        if all_rules_df.shape[0]==0:
            all_rules_df = rules_df
        else:
            all_rules_df = pd.concat([all_rules_df, rules_df])

    all_rules_df = all_rules_df.reset_index()
    if 'index' in rules_df.columns:
        all_rules_df = all_rules_df.drop(['index'], axis=1).reset_index()
        all_rules_df.rename(columns={'index': name_idx_rule}, inplace=True)

    if path:
        save_to_csv(all_rules_df, path)
    return all_rules_df


def convert_rule_to_df_old(rule, explain_bins_names):
    try:
        lht = str(rule['lhs'].iat[0])
    except:
        lht = str(rule['lhs'])
    lht = lht.replace('(', '').replace(')', '').replace("'", "").replace(" ", "").split(',')
    try:
        rht = str(rule['rhs'].iat[0])
    except:
        rht = str(rule['rhs'])
    rht = rht.replace('(', '').replace(')', '').replace("'", "").replace(" ", "").split(',')
    try:
        lst_dct = {lht[i]: lht[i + 1] for i in range(0, len(lht) - 1, 2)}
        rhs_dct = {rht[i]: rht[i + 1] for i in range(0, len(rht) - 1, 2)}
    except:
        lst_dct = {lht[0]: lht[1]}
        rhs_dct = {rht[0]: rht[1]}

    lst_dct.update(rhs_dct)
    rules_df_format = pd.DataFrame(list(lst_dct.items()), columns=['orig_column', 'bin'])
    return rules_df_format.merge(explain_bins_names, on=["orig_column", "bin"], how='left')

def convert_rule_to_df(rule, explain_bins_names):
    explain_bins_names_temp = explain_bins_names.rename({'bin': 'bin_exp', 'orig_column': 'orig_column_exp'}, axis=1)

    try:
        lht = str(rule['lhs'].iat[0])

    except:
        lht = str(rule['lhs'])
    lht = lht.replace('(', '').replace(')', '').replace("'", "").replace(" ", "").split(',')

    try:
        rht = str(rule['rhs'].iat[0])
    except:
        rht = str(rule['rhs'])
    rht = rht.replace('(', '').replace(')', '').replace("'", "").replace(" ", "").split(',')
    try:
       
        lst_dct = {lht[i]: str(lht[i + 1]) for i in range(0, len(lht) - 1, 2)}
        rhs_dct = {rht[i]: str(rht[i + 1]) for i in range(0, len(rht) - 1, 2)}

    except:
        lst_dct = {lht[0]: lht[1]}
        rhs_dct = {rht[0]: rht[1]}

    lst_dct.update(rhs_dct)
    rules_df_format = pd.DataFrame(list(lst_dct.items()), columns=['orig_column', 'bin'])
    
    rules_df_format['bin_comb'] = rules_df_format.apply(lambda x: str(x['orig_column']) + "_" + str(x['bin']), axis=1)
    if ('min' in explain_bins_names_temp.columns) and ('max' in explain_bins_names_temp.columns):
        temp = rules_df_format.merge(explain_bins_names_temp, left_on=["bin_comb"], right_on=['bin_exp'], how='left')[
            ['orig_column', 'bin', 'value', 'min', 'max']]
    elif rules_df_format.bin.dtype == 'int64':
        temp = rules_df_format.merge(explain_bins_names_temp, left_on=["bin_comb"], right_on=['bin_exp'], how='left')[
            ['orig_column', 'bin', 'value']]

    else:
        temp = rules_df_format.merge(explain_bins_names_temp, left_on=["bin"], right_on=['bin_exp'], how='left')[
            ['orig_column', 'bin', 'value']]

    return temp


def deal_with_null_value(col_of_rule, df_reduced_cols, filtered_df):
    if col_of_rule['orig_column'] in list(df_reduced_cols.select_dtypes(include=['category']).columns):
        print("{name} is of type category and can't be compared".format(name=col_of_rule['orig_column']))
        return filtered_df
    if filtered_df.shape[0] == 0:  # first update of filtered_df
        df_reduced_cols = df_reduced_cols.loc[:, ~df_reduced_cols.columns.duplicated()]
        filtered_df = df_reduced_cols[df_reduced_cols[col_of_rule['orig_column']].isna()]
    else:
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
        filtered_df = filtered_df[filtered_df[col_of_rule['orig_column']].isna()]

    return filtered_df


def filter_table_with_rule(rule, explain_bins_names, df):

    rule_df_explained = convert_rule_to_df(rule, explain_bins_names)
    filtered_df = pd.DataFrame()

    if not set(rule_df_explained['orig_column'].unique()).issubset(set(df.columns)):
        return filtered_df

    if 'index' not in df.columns:
        df = df.reset_index()
    columns_to_leave = list(rule_df_explained['orig_column'].unique()) + ['index']
    df_reduced_cols = df[columns_to_leave].copy()
    if df_reduced_cols.shape[1] < len(columns_to_leave):
        return filtered_df
    for i in range(0, rule_df_explained.shape[0]):
        col_of_rule = rule_df_explained.iloc[i]
        if '_null' in str(col_of_rule['bin']):
            filtered_df = deal_with_null_value(col_of_rule, df_reduced_cols, filtered_df.copy())
        elif str(col_of_rule['value']) != 'nan':  # check if the columns is binned
            filtered_df = deal_with_binned_column(col_of_rule, df_reduced_cols, filtered_df.copy())

        else:  # column is binary
            rule_bin = process_bin_value(col_of_rule, df)
            filtered_df = deal_with_cat_or_binary_columns(col_of_rule, df_reduced_cols, filtered_df.copy(), rule_bin)
        if filtered_df.shape[0] == 0:
            return filtered_df
    return filtered_df

def update_dict_commands(used_columns_from_sum, this_action, summary):
    command_type = this_action.action_type.iat[0]
    command_params = this_action.action_params.iat[0]
    if command_type == 'filter':
        if command_params['field'] in summary.columns:
            used_columns_from_sum['filter'] += 1
            if command_params['term'] in summary[command_params['field']].unique():
                used_columns_from_sum['value_of_filter'] += 1
    if command_type == 'group':
        if command_params['field'] in list(summary.columns):
            used_columns_from_sum['group'] += 1
            if command_params['aggregations'] != []:
                for col_agg in command_params['aggregations']:
                    if col_agg in summary[command_params['field']].unique():
                        used_columns_from_sum['group_cols_agg'] += 1
    if command_type == 'sort':
        if command_params['field'] in summary.columns:
            used_columns_from_sum['sort'] += 1
    if command_type == 'project':
        if command_params['field'] in summary.columns:
            used_columns_from_sum['project'] += 1
    return used_columns_from_sum


def calculate_all_actions(action_df):
    dict_all_actions = {'filter':0,'value_of_filter':0,'group':0,'group_cols_agg':0,'sort':0,'project':0}
    for i in range(0,action_df.shape[0]):
        this_action = action_df.iloc[i]
        if this_action.action_type=='filter':
            dict_all_actions['filter']+=1
            dict_all_actions['value_of_filter']+=1
        if this_action.action_type=='group':
            dict_all_actions['group']+=1
            if this_action.action_params['aggregations'] != []:
                for col_agg in this_action.action_params['aggregations']:
                    dict_all_actions['group_cols_agg']+=1
        if this_action.action_type=='sort':
                dict_all_actions['sort']+=1
        if this_action.action_type=='project':
                dict_all_actions['project']+=1
    return dict_all_actions
        
def project_summary_on_ar(rules, summary, explain_bins_names, min_num_rep_summary=1):
    filtered_rules = pd.DataFrame()
    for rule_idx in range(0, rules.shape[0]):
        rule = rules.iloc[rule_idx]
        rule_df = rules.iloc[[rule_idx]]
        filtered_sum = filter_table_with_rule(rule, explain_bins_names, summary)
        if filtered_sum.shape[0] >= min_num_rep_summary:
            filtered_rules = pd.concat([filtered_rules, rule_df])
    return filtered_rules


def find_coverage(rules, explain_bins_names, df):
    dict_coverage = {}
    if rules.shape[0] == 0 or len(rules) == 0:
        return dict_coverage

    if 'rule_id' not in rules:
        print("reseting rules")
        rules = rules.reset_index()
        rules.rename({'index': 'rule_id'}, inplace=True)

    if 'index' not in df.columns:
        df = df.reset_index()
    print("finding coverage for each rule in the given dataframe")
    for rule_idx in rules.index:
        rule = rules.loc[rule_idx]
        filtered_df = filter_table_with_rule(rule, explain_bins_names, df)
        if filtered_df.shape[0] > 0:
            lst_ids = list(filtered_df['index'])
            dict_coverage[rule_idx] = lst_ids
    return dict_coverage



def create_transactions(df, col_to_separate=None):
    if col_to_separate:
        transactions_dict = {}
        for bin_val in df[col_to_separate].unique():
            df_per_bin = df[df[col_to_separate] == bin_val]  # .drop([col_to_separate], axis=1)
            df_per_bin = comp_notnull1(df_per_bin)
            transaction_bin = []
            for r in df_per_bin:
                transaction_bin.append(list(r.items()))
            transactions_dict[bin_val] = transaction_bin

        return transactions_dict

    else:
        df_clean = comp_notnull1(df)
        transactions = []
        for r in df_clean:
            transactions.append(list(r.items()))

        return transactions
    
    

def create_rules_from_transactions(transactions, support_dict, confidence_dict, path=None):
    """

    Args:
        transactions: needs to be either a list in the goal column wasn't defined or a dictionary if it was
        support:
        confidence:
        path:

    Returns:

    """
    if type(transactions) == dict:  # we got a goal column
        rules_dict = {}
        for bin_value in transactions.keys():
            transaction = transactions[bin_value]
            itemsets, rules = apriori(transaction, min_support=support_dict[bin_value],
                                      min_confidence=confidence_dict[bin_value])
            attrs = [a for a in dir(rules[0]) if not a.startswith("_")]
            rules_rec = []
            for r in rules:
                rdict = {}
                for a in attrs:
                    rdict[a] = getattr(r, a)
                    rdict["rule"] = str(r).split("} (")[0] + "}"
                    rdict["len_l"] = len(r.lhs)
                    rdict["len_r"] = len(r.rhs)
                rules_rec.append(rdict)

            rules_df = pd.DataFrame(rules_rec)
            rules_df.set_index('rule', inplace=True)
            selected_columns = ['len_l', 'len_r', 'count_lhs', 'count_rhs', 'support', 'confidence', 'lift', 'rpf',
                                'conviction', 'lhs',
                                'rhs']
            rules_df = rules_df[selected_columns]
            rules_dict[bin_value] = rules_df
        return rules_dict

    else:
        itemsets, rules = apriori(transactions, min_support=support_dict, min_confidence=confidence_dict)
        attrs = [a for a in dir(rules[0]) if not a.startswith("_")]
        rules_rec = []
        for r in rules:
            rdict = {}
            for a in attrs:
                rdict[a] = getattr(r, a)
                rdict["rule"] = str(r).split("} (")[0] + "}"
                rdict["len_l"] = len(r.lhs)
                rdict["len_r"] = len(r.rhs)
            rules_rec.append(rdict)

        rules_df = pd.DataFrame(rules_rec)
        rules_df.set_index('rule', inplace=True)
        selected_columns = ['len_l', 'len_r', 'count_lhs', 'count_rhs', 'support', 'confidence', 'lift', 'rpf',
                            'conviction', 'lhs',
                            'rhs']
        rules_df = rules_df[selected_columns]
        if path:
            save_to_csv(rules_df, path)

        return rules_df




def pick_random_representative_from_rules(filtered_rules, explain_bins_names, df, enforce_size=False,num_rows=10,num_cols=10,config_params=None):
    list_idx = df['index'].to_list()
    created_summary = pd.DataFrame()
    selected_columns = []
    for rule_idx in range(0, filtered_rules.shape[0]):
        rule = filtered_rules.iloc[rule_idx]
        rule_df_explained = convert_rule_to_df(rule, explain_bins_names)

        # extraction of columns to be selected
        for i in range(0, rule_df_explained.shape[0]):
            part_rule_explained = rule_df_explained.iloc[i]
            part_rule_col_name = part_rule_explained['orig_column']
            selected_columns.append(part_rule_col_name)

        candidates = list(filter_table_with_rule(rule, explain_bins_names, df).index)
        candidates_filtered = list(np.intersect1d(list_idx, candidates))
        if len(candidates_filtered) >= 1:
            rand_row = sample(candidates_filtered, 1)[0]
            temp_df = pd.DataFrame()
            temp_df = df[df['index'] == rand_row]
            created_summary = pd.concat([created_summary, temp_df])
        else:
            print("problem with rule:")
            print(rule_df_explained)

    if enforce_size:
        while created_summary.shape[0] < num_rows:
            rule = filtered_rules.iloc[sample(range(0, filtered_rules.shape[0]), k=1)]
            rule_df_explained = convert_rule_to_df(rule, explain_bins_names)

            # extraction of columns to be selected
            for i in range(0, rule_df_explained.shape[0]):
                part_rule_explained = rule_df_explained.iloc[i]
                part_rule_col_name = part_rule_explained['orig_column']
                selected_columns.append(part_rule_col_name)

            candidates = list(filter_table_with_rule(rule, explain_bins_names, df).index)
            candidates_filtered = list(np.intersect1d(list_idx, candidates))
            if len(candidates_filtered) >= 1:
                rand_row = sample(candidates_filtered, 1)[0]
                temp_df = pd.DataFrame()
                temp_df = df[df['index'] == rand_row]
                created_summary = pd.concat([created_summary, temp_df])
            else:
                print("problem with rule:")
                print(rule_df_explained)
        if config_params:
            selected_columns_reduced = list(set(selected_columns + config_params['OBJ_COLS'] + config_params['KEY_COLS']))
        else:
            selected_columns_reduced = list(set(selected_columns))
        if config_params:
            if len(selected_columns_reduced) < num_cols + len(
                    config_params['OBJ_COLS'] + config_params['KEY_COLS']):
                additional_columns = num_cols - len(selected_columns_reduced)
                remaining_cols = list(set(np.setdiff1d(df.columns, selected_columns_reduced)))

                t = sample(remaining_cols, k=additional_columns)
                selected_columns_reduced += t
        else:
            if len(selected_columns_reduced) < num_cols:
                additional_columns = num_cols - len(selected_columns_reduced)
                remaining_cols = list(set(np.setdiff1d(df.columns, selected_columns_reduced)))

                t = sample(remaining_cols, k=additional_columns)
                selected_columns_reduced += t
    else:
        selected_columns_reduced = set(selected_columns)

    created_summary = created_summary[selected_columns_reduced]
    return created_summary



def pick_row_from_rule(rule, explain_bins_names, df):
    list_idx = df['index'].to_list()
    selected_columns = []

    rule_df_explained = convert_rule_to_df(rule, explain_bins_names)

    # extraction of columns to be selected
    for i in range(0, rule_df_explained.shape[0]):
        part_rule_explained = rule_df_explained.iloc[i]
        part_rule_col_name = part_rule_explained['orig_column']
        selected_columns.append(part_rule_col_name)

    candidates = list(filter_table_with_rule(rule, explain_bins_names, df).index)
    candidates_filtered = list(np.intersect1d(list_idx, candidates))
    if len(candidates_filtered) >= 1:
        rand_row = sample(candidates_filtered, 1)[0]

    else:
        print("problem with rule:")
        print(rule_df_explained)

    return rand_row, selected_columns


def display_as_itemset(rules,col_name='rule',col_name_itemset='itemset'):
    rules[col_name] = rules.apply(
        lambda x: str(x[col_name]).replace('} -> {', ', ').replace("}", "") + ", (" + str(x.rhs).split('),(')[-1].replace(
            '))', ')}'),
        axis=1)
    rules[col_name_itemset] = rules[col_name].apply(lambda x: str(list(set([i.replace("{", "").replace("(", "").replace("}",
                                                                                                                    "").replace(
        "),", "").replace(")", "").replace(" ", "").replace("'", "") for i in str(x).split('), (')]))))
    rules = rules.drop_duplicates(subset=col_name_itemset)

    return rules


def find_rules_from_df1_without_df2(df1, df2, col_to_compare, col_idx='rule_id'):
    rules_unique = []
    for val in df1[col_to_compare]:
        if df2[df2[col_to_compare] == val].shape[0] == 0:
            rules_unique.append(df1[df1[col_to_compare] == val][col_idx].iat[0])
    return df1[df1[col_idx].isin(rules_unique)]


def remove_redundant_itemset(df, col_name='rule', col_name_itemset='itemset'):
    idx = 0
    for itemset in df[col_name]:
        lst_items = itemset.split('),')
        sorted_list = []
        for item in lst_items:
            temp = item.replace("{", "").replace("(", "").replace("}", "").replace(")", "").replace(" ", "")
            sorted_list.append(temp)
        df.loc[idx, col_name_itemset] = str(sorted(sorted_list))
        idx += 1
    return df.drop_duplicates(col_name_itemset)


def pick_rules_greedy_advanced(dict_rules, rules_df, full_df, mapping_bin_values, num_rules=10):
    max_rule_idx = list(dict_rules.keys())[0]
    chosen_rules = [max_rule_idx]
    for i in range(1, num_rules):
        max_score_so_far = 0
        candidate = None
        for rule_idx in list(dict_rules.keys()):
            if rule_idx not in chosen_rules:
                rules_considered_now_df = rules_df[rules_df.index.isin(chosen_rules + [rule_idx])]
                cell_score, sum_df = cell_coverage(rules_considered_now_df, full_df, mapping_bin_values)
                if cell_score > max_score_so_far:
                    max_score_so_far = cell_score
                    candidate = rule_idx
        chosen_rules.append(candidate)
    return chosen_rules



def find_closest_row(c, vec_list):
    if type(vec_list) == pd.core.frame.DataFrame:
        vec_list = vec_list.to_numpy()
    closest_vec = vec_list[0]

    idx = 0
    min_distance = np.abs(euclidean_distances(c.reshape(1, -1), closest_vec.reshape(1, -1))[0][
                              0])  # reshaping means we have only one sample in each array.
    for i, vec in enumerate(vec_list):
        distance = np.abs(euclidean_distances(c.reshape(1, -1), vec.reshape(1, -1))[0][0])
        if distance < min_distance:
            closest_vec = vec
            min_distance = distance
            idx = i

    return idx, closest_vec

def transform_single_value_to_dict(confidence_dict, support_dict, df_col_to_separate_bins):
    """
    If the confidence/support is the same for each bin value, we create a dictionary for each bin with the given confidence/support
    Args:
        confidence_dict: single value or dict with value per bin
        support_dict: single value or dict with value per bin
        df_col_to_separate_bins: the different bin in the goal column

    Returns: transofrmed confidence and support

    """
    if type(confidence_dict) == float:
        new_confidence_dict = {}
        for bin_val in df_col_to_separate_bins:
            new_confidence_dict[bin_val] = confidence_dict
        confidence_dict = new_confidence_dict
    if type(support_dict) == float:
        new_support_dict = {}
        for bin_val in df_col_to_separate_bins:
            new_support_dict[bin_val] = support_dict
        support_dict = new_support_dict
    return confidence_dict, support_dict

def create_association_rules(df, prefix_ar, support_dict=0.1, confidence_dict=0.05, name_idx_rule='rule_id',
                             col_to_separate=None, print_results=False):
    """

    Args:
        print_results: If True, prints the progress
        df: The original data (binned)
        prefix_ar: path to store the rules
        support_dict: could be either a dict - different support per bin {0:0.1} or a single value in case col_to_separate=None
        confidence_dict: could be either a dict - different confidence per bin {0:0.1} or a single value in case col_to_separate=None
        name_idx_rule: The name of the index of rules dataset
        col_to_separate:  the input should be a string with the name of the column

    Returns: Association rules

    """

    path_concat_rules = prefix_ar + '/rules.csv'
    path_list = prefix_ar + '/sup_{sup}_con_{con}_list.csv'.format(sup=str(support_dict), con=str(confidence_dict))
    if print_results:
        print("started computing association rules")

    if col_to_separate is not None:
        if type(col_to_separate)==list and len(col_to_separate)==1 and col_to_separate[0] in list(df.columns):
            col_to_separate = col_to_separate[0]
        elif type(col_to_separate)==str and col_to_separate in list(df.columns):
            col_to_separate = col_to_separate
        else:
            col_to_separate =None

    if col_to_separate is not None:
        if print_results:
            print("computing association rules with goal column")

        confidence_dict, support_dict = transform_single_value_to_dict(confidence_dict, support_dict,
                                                                       df_col_to_separate_bins=df[col_to_separate].unique())
        list_of_transactions = create_transactions(df, col_to_separate=col_to_separate)
        if print_results:
            print("finished creating transactions")

        rules_per_group = create_rules_from_transactions(transactions=list_of_transactions, support_dict=support_dict,
                                                         confidence_dict=confidence_dict, path=path_list)

        if print_results:
            print("finished creating rules per value of the goal colunm")
        rules = concat_and_filter_rules(rules_per_group, path=path_concat_rules)
        if print_results:
            print("finished final filtering of the rules")
    else:    
        if print_results:
            print("computing association rules without goal column")
        transactions = create_transactions(df, col_to_separate=col_to_separate)
        if print_results:
            print("finished creating transactions")

        rules = create_rules_from_transactions(transactions=transactions, support_dict=support_dict, \
                                               confidence_dict=confidence_dict, path=path_list)
        if print_results:
            print("finished creating rules")
        rules = rules[rules['support'] > support_dict]
        rules = rules.reset_index()
        if 'index' in rules.columns:
            rules = rules.drop(['index'], axis=1)
        else:
            rules = rules.reset_index()
        rules.rename(columns={'index': name_idx_rule}, inplace=True)

    if print_results:
        print("transforming to itemset and removing redundant rules")
    rules_itemset = display_as_itemset(rules, col_name='rule', col_name_itemset='itemset')
    if name_idx_rule not in rules_itemset.columns:
        rules_itemset = rules_itemset.reset_index().rename({'index': name_idx_rule}, axis=1)
    else:
        rules_itemset = rules_itemset.reset_index().drop('index',axis=1)
    return rules_itemset


def create_dict_rules_of_each_label(dataset, rules_itemset_uni, mapping_bin_values, rules, THRESHOLD_LABELS=0):
    rules_of_each_label = defaultdict(list)
    labels_list = list(dataset['label'].unique())
    for label in labels_list:
        label_table = dataset[dataset['label'] == label].drop('label', axis=1)
        for rule_id in list(rules_itemset_uni['rule_id']):
            rule = rules[rules['rule_id'] == rule_id]
            filtered_df = filter_table_with_rule(rule, mapping_bin_values, label_table)
            if filtered_df.shape[0] > THRESHOLD_LABELS:
                rules_of_each_label[label].append(rule['itemset'].iat[0])
    return rules_of_each_label


def create_heatmap(labels_list, rules_of_each_label):
    df_full = pd.DataFrame(columns=labels_list)
    for label_row in labels_list:
        dict_label = defaultdict(float)
        for label_col in labels_list:
            mutual_elements = len(
                list(set(rules_of_each_label[label_row]).intersection(rules_of_each_label[label_col])))
            num_elements_two_lists = len(set(rules_of_each_label[label_row] + rules_of_each_label[label_col]))
            if num_elements_two_lists != 0:
                dict_label[label_col] = [np.round(mutual_elements / (num_elements_two_lists + 0.00001), 2)]
            else:
                dict_label[label_col] = 0

        df_label = pd.DataFrame.from_dict(dict_label)
        df_full = pd.concat([df_full, df_label])

    df_full.index = (list(df_full.columns))
    return df_full


def compute_difference_of_two_summaries(summary_1, summary_2, full_dataset):
    difs_dict: defaultdict[Any, float] = defaultdict(float)

    for col in list(summary_1.columns):
        range_of_col = full_dataset[col].max() - full_dataset[col].min() + 1

        for cell in summary_1[col]:
            difs_dict[col] += abs((cell - min(summary_2[col], key=lambda x: abs(x - cell))) / range_of_col)

    return difs_dict


def clean_summary(summary, cols_to_drop=['index', 'Unnamed: 0'], indicate_for_cols_to_drop=['.'], num_cols=[9, 11],
                  print_size_warning=False, baseline=None):
    for col_to_drop in cols_to_drop:
        if col_to_drop in summary.columns:
            summary.drop([col_to_drop], axis=1, inplace=True)
    if list(set(summary.columns)) != list(summary.columns):
        summary = summary.iloc[:, ~summary.columns.duplicated()]
        for col in summary.columns:
            for word in indicate_for_cols_to_drop:
                if word in col:
                    summary.drop(col, axis=1, inplace=True)

    if print_size_warning and baseline:
        if summary.shape[1] < num_cols[0] or summary.shape[1] > num_cols[1]:
            print("need to fix summary {}".format(baseline))
            print(list(summary.columns))

    return summary


def create_binned_summary(summary, mapping_bin_values):
    summary_binned = pd.DataFrame()
    columns_mapped = list(mapping_bin_values['orig_column'].unique())
    for i in range(0, summary.shape[0]):
        row = defaultdict()
        for j in range(0, summary.shape[1]):
            col = summary.columns[j]
            value = summary.iloc[i, j]
            if col not in columns_mapped:
                binned_value = col + str(value)
            else:
                if mapping_bin_values[
                    (mapping_bin_values['orig_column'] == col) & (value > mapping_bin_values['min']) & (
                            value <= mapping_bin_values['max'])]['bin'].shape[0] > 0:
                    binned_value = mapping_bin_values[
                        (mapping_bin_values['orig_column'] == col) & (value > mapping_bin_values['min']) & (
                                    value <= mapping_bin_values['max'])]['bin'].iat[0]
                else:
                    binned_value = col + '_' + str(value)
            row[col] = [binned_value]
        row_df = pd.DataFrame.from_dict(row)
        summary_binned = pd.concat([row_df, summary_binned])

    return summary_binned





def deal_with_binned_column(col_of_rule, df_reduced_cols, filtered_df):
    clean_range = str(col_of_rule['value']).replace('(', '').replace(')', '').replace(']', '').split(',')
    min_val = float(clean_range[0])
    max_val = float(clean_range[1])

    if col_of_rule['orig_column'] in list(df_reduced_cols.select_dtypes(include=['category']).columns):
        print("{name} is of type category and can't be compared".format(name=col_of_rule['orig_column']))
        return filtered_df
    if filtered_df.shape[0] == 0:  # first update of filtered_df
        df_reduced_cols = df_reduced_cols.loc[:, ~df_reduced_cols.columns.duplicated()]
        filtered_df = df_reduced_cols[((df_reduced_cols[col_of_rule['orig_column']] > min_val) & (
                df_reduced_cols[col_of_rule['orig_column']] <= max_val))].copy()
    else:
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]

        filtered_df = filtered_df[
            ((filtered_df[col_of_rule['orig_column']] > min_val) & (
                    filtered_df[col_of_rule['orig_column']] <= max_val))].copy()

    return filtered_df


def deal_with_cat_or_binary_columns(col_of_rule, df_reduced_cols, filtered_df, rule_bin):
    if filtered_df.shape[0] == 0:  # creation of filtered_df

        df_reduced_cols = df_reduced_cols.loc[:, ~df_reduced_cols.columns.duplicated()]
        filtered_df = df_reduced_cols[(df_reduced_cols[col_of_rule['orig_column']] == rule_bin)].copy()

    else:  # update of filtered_df

        filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
        filtered_df = filtered_df[(filtered_df[col_of_rule['orig_column']] == rule_bin)].copy()
    return filtered_df




def process_bin_value(col_of_rule, df):

    if df[col_of_rule['orig_column']].dtype == 'int64':
        if type(col_of_rule['bin']) == np.int64:
            rule_bin = int(col_of_rule['bin'])

        elif "_" in col_of_rule['bin']:
            rule_bin = col_of_rule['bin']

        else:
            rule_bin = int(col_of_rule['bin'])
    else:
        rule_bin = col_of_rule['bin']
    if rule_bin == 'False' or rule_bin == '0':
        rule_bin = 0
    elif rule_bin == 'True' or rule_bin == '1':
        rule_bin = 1
    return rule_bin



def color_and_hover_summary(df, rule_dict):
    df_cols = list(df.columns)
    def style_spec(x):
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        for r_dict in rule_dict:
            for idx_rule in r_dict['rule_index']:
                df1.at[idx_rule[0], df_cols[idx_rule[1]]] = r_dict['color']
        return df1

    def create_tooltips_df(x):
        
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        for r_dict in rule_dict:
            for idx_rule in r_dict['rule_index']:
                df1.at[idx_rule[0], df_cols[idx_rule[1]]] = r_dict['rule_text']
                
        return df1
    
    display(df.style.apply(style_spec, axis=None).format(precision=2).set_tooltips(create_tooltips_df(df)))
