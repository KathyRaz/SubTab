import ast
import os
import random
import uuid
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd
from IPython.display import display

from associations_rules_summary.utils_code import color_and_hover_summary, project_summary_on_ar
from word2vec_embedding.utils_w2v import data_transformation, \
    create_tab_vec_with_emb, create_summary, create_summary_for_filtered_dataset_memory

BASE_DIR = "models/"


def gen_model_uuid(label=None):
    uu = uuid.uuid4().hex
    # os.path.join(
    # os.mkdir(uu)
    if label:
        return f"{uu}_{label}"
    else:
        return uu


def gen_dir(model_uuid, added, base_dir='None'):
    #    base_dir=utils.config.BASE_DIR
    if base_dir == 'None':
        base_dir = BASE_DIR

    new_dir = os.path.join(base_dir, model_uuid, added)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    return new_dir


def create_filtered_rules_per_row(summary, rules, mapping_bins_to_values):
    all_filtered_rules = {}
    all_filtered_rules_length = {}
    for row_id in summary.index:
        this_row = summary.loc[[row_id]]
        filtered_rules = project_summary_on_ar(rules, this_row, mapping_bins_to_values)
        if filtered_rules.shape[0] > 0:
            filtered_rules['filtered_ruleslen_itemset'] = filtered_rules['itemset'].apply(lambda x: len(x.split(',')))
            filtered_rules = filtered_rules.sort_values('filtered_ruleslen_itemset', ascending=False)
            all_filtered_rules[row_id] = filtered_rules
            all_filtered_rules_length[row_id] = filtered_rules.shape[0]
            all_filtered_rules_length = dict(
                sorted(all_filtered_rules_length.items(), key=lambda item: item[1], reverse=True))

    return all_filtered_rules, all_filtered_rules_length


def create_rule_text(rule):
    # print("create_rule_text")
    lhs = rule.lhs.iat[0]
    if type(lhs) == str:
        lhs = ast.literal_eval(lhs)

    rhs = rule.rhs.iat[0]
    if type(rhs) == str:
        rhs = ast.literal_eval(rhs)

    rule_text = 'Rule #{}: {{'.format(rule.index[0])
    for e in lhs:
        try:
            rule_text = rule_text + str("({e1} = {e2}),".format(e1=e[0], e2=e[1]))
        except:
            print("error occured in lhs in rule: ", str(rule.index[0]))
            print(e)
            print(lhs)
            rule_text = rule_text + str("({}),".format(str(e)))

    rule_text = rule_text + str("} ---> {")
    for e in rhs:
        try:
            rule_text = rule_text + str("({e1} = {e2}),".format(e1=e[0], e2=e[1]))
        except:
            print("error occured in rhs in rule: ", str(rule.index[0]))
            print(e)
            print(rhs)
            rule_text = rule_text + str("({}),".format(str(e)))

    rule_text = rule_text + str("}, ")
    rule_text = rule_text + str(
        f"support={round(rule.support.iat[0], 2)}, confidence = {round(rule.confidence.iat[0], 2)}, lift = {round(rule.lift.iat[0], 2)}")
    return rule_text


def create_rule_index(rule, row_id, order_columns):
    # print("create_rule_index")

    lhs = rule.lhs.iat[0]
    if type(lhs) == str:
        lhs = ast.literal_eval(lhs)

    rhs = rule.rhs.iat[0]
    if type(rhs) == str:
        rhs = ast.literal_eval(rhs)
    rule_index = []
    for item in lhs:
        try:
            rule_index.append([row_id, order_columns.index(item[0])])
        except:
            print("error at lhs")
            print(item)
    for item in rhs:
        try:
            rule_index.append([row_id, order_columns.index(item[0])])
        except:
            print("error at lhs")

            print(item)
    return rule_index


def create_dict_patterns(summary, rules, mapping_bins_to_values, df):
    from metrics import jaccard_metric, cell_coverage_counting
    # print("create dict patterns")
    possible_colors = ['green', 'blue', 'yellow', 'red', 'orange', 'pink', 'lightblue', 'purple', 'gray', 'darkgrey']
    order_columns = list(summary.columns)
    filtered_rules = project_summary_on_ar(rules.copy(), summary.copy(), mapping_bins_to_values.copy())
    cell_cov = cell_coverage_counting(filtered_rules.copy(), df.copy(), mapping_bins_to_values.copy())
    jac_score = round(1 - jaccard_metric(summary=summary.copy(), mapping_bin_values=mapping_bins_to_values.copy(),
                                         bin_transform=True), 2)

    used_rules = []
    rule_dict = []

    all_filtered_rules, all_filtered_rules_length = create_filtered_rules_per_row(summary.copy(), rules.copy(),
                                                                                  mapping_bins_to_values.copy())

    for row_id in all_filtered_rules_length.keys():
        rule_for_row = 0
        for rule_id in all_filtered_rules[row_id].rule_id.unique():
            if rule_for_row == 0:
                if rule_id not in used_rules:
                    used_rules.append(rule_id)
                    rule_for_row = 1
                    rule = all_filtered_rules[row_id][all_filtered_rules[row_id].rule_id == rule_id]
                    rule_text = create_rule_text(rule)
                    rule_index = create_rule_index(rule, row_id, order_columns)
                    rule_color = random.choice(possible_colors)
                    possible_colors.remove(rule_color)
                    rule_dict.append({'rule_text': rule_text, 'rule_index': rule_index,
                                      'color': 'background-color : ' + str(rule_color)})

        if rule_for_row == 0:
            print("couldn't find matching rule for row number ", row_id)

    dict_patterns = {'order_columns': list(summary.columns),
                     'idx_summary': list(summary.index),
                     'rule_dict': rule_dict,
                     'dict_eval': {'cell_cov': cell_cov, 'jaccard': jac_score}}

    return dict_patterns


class SubTub:
    """
    the class that holds SubTub instance. It load a dataframe, creates a binning transformation, and then creates
    embedding based on the requested method. it stores the columns and rows vectors. it has a function of display to
    another dataframe (after exploration), and could be given location to association rules for evaluation and
    visualization.
    """

    def __init__(self, df, use_rules=False, subtub_config={}):

        self.subtub_config = subtub_config
        self.df = df

        if 'NUM_COLS' in subtub_config.keys():
            self.num_cols = subtub_config['NUM_COLS']
        else:
            self.num_cols = 10
        if 'NUM_ROWS' in subtub_config.keys():
            self.num_rows = subtub_config['NUM_ROWS']
        else:
            self.num_rows = 10

        self.orig_df = df

        if 'prefix' in subtub_config.keys():
            self.prefix = subtub_config['prefix']
        else:
            self.prefix = os.getcwd()

        if 'name_dataset' in subtub_config.keys():
            self.name_dataset = subtub_config['name_dataset']

        else:
            self.name_dataset = 'users_dataset'

        if 'nulls_ratio' in subtub_config.keys():
            self.null_ratio = subtub_config['nulls_ratio']
            if self.null_ratio > 0:
                self.take_nulls = True

        else:
            self.null_ratio = 0
            self.take_nulls = False

        self.model_description = "{}_binned_data_cell_to_vec".format(self.name_dataset)

        if 'MODEL_UUID' in subtub_config.keys():
            self.model_uuid = subtub_config['MODEL_UUID']
            subtub_config['VECTORIZE_TABLES'] = 'False'
            subtub_config['BUILD_CORPUS'] = 'False'
            subtub_config['BUILD_MODEL'] = 'False'
        else:
            self.model_uuid = gen_model_uuid(self.model_description)
            subtub_config['VECTORIZE_TABLES'] = 'True'
            subtub_config['BUILD_CORPUS'] = 'True'
            subtub_config['BUILD_MODEL'] = 'True'

        if 'CONF' not in subtub_config.keys():
            subtub_config['CONF'] = {'add_attr': 'True',
                                     'add_columns': 'True',
                                     'vector_size': 50,
                                     'row_window_size': 10,
                                     'col_window_size': 100,
                                     'min_count': 2}
        if use_rules:
            if 'rules' in subtub_config.keys():
                try:
                    self.rules = pd.read_csv(subtub_config['rules'])
                    # self.mapping_bins_to_values = pd.read_csv(subtub_config['mapping_bins_to_values'])
                except:
                    self.rules = None
                    print("the location of provided rules or mapping_bins_to_values is incorrect")
            else:
                self.rules = None
                print("print no rules were provided")
        # binning
        self.binned_df, self.full_df, self.mapping_bins_to_values = data_transformation(df,
                                                                                        MODEL_UUID=self.model_uuid,
                                                                                        name_dataset=self.name_dataset,
                                                                                        prefix=self.prefix,
                                                                                        q=5,
                                                                                        precision=0,
                                                                                        path_df_binned=None,
                                                                                        trim_multi_val_col=True,
                                                                                        create_mapping=True,
                                                                                        path_mapping=None,
                                                                                        config_params=None,
                                                                                        nulls_ratio=self.null_ratio,
                                                                                        binning_method='custom',
                                                                                        print_time=True)

        # embbeding:

        self.vec_list_rows, self.vec_list_cols, self.model, self.corpus_tuple = create_tab_vec_with_emb(
            dataset=self.binned_df,
            config_params=subtub_config,
            CONF=subtub_config['CONF'],
            MODEL_UUID=self.model_uuid,
            name_dataset=self.name_dataset, prefix=self.prefix)

    def display(self, df, clustering_algo='KMeans'):
        start_cols_rows = timer()

        if df.equals(self.df):
            # print("equals")
            exp_summary = create_summary(df, self.vec_list_rows, vec_list_w2v_cols=self.vec_list_cols,
                                         clustering_algo=clustering_algo, n_clusters=self.num_cols)
        else:
            a = df.index
            b = self.full_df.index
            df = self.full_df.loc[list(set(b).intersection(set(a)))]
            exp_summary = create_summary_for_filtered_dataset_memory(filtered_df=df, clustering_algo=clustering_algo,
                                                                     n_clusters=self.num_cols, dataset=self.binned_df,
                                                                     cell_dict=self.corpus_tuple[2],
                                                                     take_nulls=self.take_nulls,
                                                                     w2v_model=self.model)

        if self.rules is not None:
            dict_patterns = create_dict_patterns(exp_summary, self.rules, self.mapping_bins_to_values, df)
            dict_eval = dict_patterns['dict_eval']
            rule_dict = dict_patterns['rule_dict']
            display(dict_eval)
            end_cols_rows = timer()
            print("for or summary creation, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))

            df1 = color_and_hover_summary(exp_summary[dict_patterns['order_columns']].round(2),
                                          dict_patterns['rule_dict'])

            # display(df1)

        else:
            end_cols_rows = timer()
            print("for or summary creation, it took {}".format(timedelta(seconds=end_cols_rows - start_cols_rows)))
            display(exp_summary.round(2))
