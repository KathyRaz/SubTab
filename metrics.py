from sklearn.metrics import jaccard_score
from scipy.spatial import distance
from math import log2

from associations_rules_summary.utils_code import *
from Random_baselines.random_util import *


def evaluate_summary(summary, rules, mapping_bin_values, full_df, metrics=None):
    if metrics is None:
        metrics = ['naive', 'naive_norm', 'cell_cov', 'num_rules', 'num_bins', 'jaccard']  # 'covariance','cross-ent']
    dict_scores = {}
    filtered_rules = project_summary_on_ar(rules, summary, mapping_bin_values)
    print("filtered rules of size:")
    print(filtered_rules.shape)
    dict_coverage = find_coverage(filtered_rules, mapping_bin_values, full_df)
    for metric in metrics:
        if metric == 'naive':
            naive_score = naive_score_coverage(dict_coverage, rules, full_df.shape[0])
            dict_scores['naive'] = [naive_score]
        elif metric == 'naive_norm':
            norm_score, rules_w_sup = normalized_score_coverage(dict_coverage, rules, full_df.shape[0])
            dict_scores['naive_norm'] = [norm_score]
        elif metric == 'cell_cov':
            try:
                cell_score = cell_coverage_counting(filtered_rules, full_df, mapping_bin_values)
                dict_scores['cell_cov'] = [cell_score]
            except:
                dict_scores['cell_cov'] = [None]
        elif metric == 'num_rules':
            num_rules_score = round(filtered_rules.shape[0] / rules.shape[0], 2)
            dict_scores['num_rules'] = [num_rules_score]
        elif metric == 'num_bins':
            try:
                interestigness_score, bins_represented = interestingness_bins_count_score(mapping_bin_values, full_df,
                                                                                          summary)
                dict_scores['num_bins'] = [interestigness_score]
            except:
                dict_scores['num_bins'] = [None]
        elif metric == 'covariance':
            dict_scores['covariance'] = covariance_metric(summary, full_df)
        elif metric == 'cross-ent':
            dict_scores['cross-ent'] = cross_entropy_metric(summary, full_df)
        elif metric == 'jaccard':
            dict_scores['jaccard'] = [
                jaccard_metric(summary=summary, mapping_bin_values=mapping_bin_values, bin_transform=True)]
    return dict_scores


def naive_score_coverage(dict_coverage, rules, df_total_num_rows):
    set_index = set()
    for rule_num in dict_coverage.keys():
        set_index = set_index.union(set(dict_coverage[rule_num]))
    return round(len(set_index) / df_total_num_rows, ndigits=2)


def cell_coverage(filtered_rules, df, explain_bins_names, rounding_gran: int = 3):
    if len(filtered_rules) == 0 or filtered_rules.shape[0] == 0:
        return 0, pd.DataFrame()

    summarizing_df = pd.DataFrame(index=df.index, columns=df.columns).fillna(0)
    for i in tqdm(range(0, filtered_rules.shape[0])):
        a = filtered_rules.iloc[i]
        rule_df_explained = convert_rule_to_df(a, explain_bins_names)
        a_applied = filter_table_with_rule(a, explain_bins_names, df)[list(rule_df_explained.orig_column)]
        cols_to_look = a_applied.columns
        for a_col in cols_to_look:
            a_applied[a_col] = 1
        temp = a_applied.merge(summarizing_df, how='right', left_index=True, right_index=True).fillna(0)
        temp_new = temp.copy()

        for col_to_look in cols_to_look:
            cols_similar = []
            for col in temp_new.columns:
                if (col_to_look + "_x" in col) or (col_to_look + "_y" in col):
                    cols_similar.append(col)
            temp_new[col_to_look] = temp[cols_similar[0]] + temp[cols_similar[1]]
            temp_new.drop(cols_similar, axis=1, inplace=True)

        summarizing_df = temp_new.copy()

    score = 0
    max_score = summarizing_df.shape[0] * summarizing_df.shape[1]
    for col in summarizing_df.columns:
        score += summarizing_df[summarizing_df[col] == 0].shape[0]
    final_score = round(1 - score / max_score, rounding_gran)
    return final_score, summarizing_df


def cell_coverage_counting(filtered_rules, df, explain_bins_names, rounding_gran: int = 3, print_progress=False):
    counting_cells_per_columns = {}

    for i in tqdm(range(0, filtered_rules.shape[0])):
        a = filtered_rules.iloc[i]
        rule_df_explained = convert_rule_to_df(a, explain_bins_names)
        a_applied = filter_table_with_rule(a, explain_bins_names, df)[list(rule_df_explained.orig_column)]
        if a_applied.shape[0] > 0:
            for col in a_applied.columns:
                if col in counting_cells_per_columns.keys():
                    counting_cells_per_columns[col].extend(list(a_applied.index))
                    counting_cells_per_columns[col] = list(set(counting_cells_per_columns[col]))
                else:
                    counting_cells_per_columns[col] = list(a_applied.index)
    score = 0
    max_score = df.shape[0] * df.shape[1]
    for col in counting_cells_per_columns.keys():
        score += len(counting_cells_per_columns[col])
    final_score = round(score / max_score, rounding_gran)

    return final_score


def normalized_score_coverage(dict_coverage, rules, df_total_num_rows):
    """
    # sort the rules, from the longest to the shortest
    # for each row that the rule applies to assign score of {size_of_rule}
    # row that got a score will no longer get scores.
    # output the cumulative score of all the rows that have been scored
    """
    if len(rules) == 0:
        print("rules are empty")
        return 0, pd.DataFrame(columns=["rule_id", "num_rows_sup", "cumulative_score", "prev_score", "delta_score"])
    set_index = set()
    score = 0
    # if 'rule_id' not in rules:
    #     rules_updated = rules.reset_index().copy()
    #     print("rested index")
    # else:
    rules_updated = rules.copy()

    rules_sorted = list(rules_updated.sort_values('len_l', ascending=False).index)
    max_sum = 0
    # find maximum length of a rule:
    for rule_num in tqdm(rules_updated.index):
        current_sum = rules_updated.loc[rule_num, 'len_l'] + rules_updated.loc[rule_num, 'len_r']
        if current_sum > max_sum:
            max_sum = current_sum

    for rule_num in tqdm(rules_sorted):
        if rule_num in dict_coverage.keys():
            rows_to_score = []
            row_numbers = set(dict_coverage[rule_num])
            for selected_row in row_numbers:
                if selected_row not in set_index:
                    rows_to_score.append(selected_row)

            prev_score = score
            score += (rules_updated.loc[rule_num, 'len_l'] + rules_updated.loc[rule_num, 'len_r']) * len(rows_to_score)
            rules_updated.loc[rule_num, 'num_rows_sup'] = len(rows_to_score)
            rules_updated.loc[rule_num, 'cumulative_score'] = score
            rules_updated.loc[rule_num, 'prev_score'] = prev_score
            rules_updated.loc[rule_num, 'delta_score'] = score - prev_score
            set_index = set_index.union(row_numbers)
    div_factor = df_total_num_rows * max_sum
    final_score = round(score / div_factor, ndigits=2)
    if final_score == 0:
        return 0, pd.DataFrame(columns=["rule_id", "num_rows_sup", "cumulative_score", "prev_score", "delta_score"])
    return final_score, rules_updated


def interestingness_bins_count_score(mapping_bin_values, full_df, summary):
    bins_represented = []
    interestigness_score = 0
    num_cols = len(full_df.columns)
    for col in summary.columns:
        map_df = mapping_bin_values[mapping_bin_values['orig_column'] == col][['bin', 'value']]
        unique_vals = list(summary[col].unique())
        bins_represented_col = []

        if map_df.shape[0] > 0:  # column was binned:
            bin_value_col = map_df['value']
            num_bins = bin_value_col.shape[0]  # all possible bins
            for bin_interval_idx in range(0, bin_value_col.shape[0]):
                bin_name = map_df['bin'].iat[bin_interval_idx]
                if '_null' not in bin_name:
                    temp = str(bin_value_col.iat[bin_interval_idx]).replace('(', '') \
                        .replace(']', '').replace(' ', '').split(',')
                    iv = pd.Interval(left=float(temp[0]), right=float(temp[1]), closed='right')
                    for uni_val in unique_vals:
                        if uni_val in iv:
                            if map_df['bin'].iat[bin_interval_idx] not in bins_represented_col:
                                bins_represented_col.append(bin_name)
                else:
                    if summary[summary[col].isna()].shape[0] > 0 and bin_name not in bins_represented_col:
                        bins_represented_col.append(bin_name)

        else:
            num_bins = full_df[col].nunique()
            for uni_val in unique_vals:
                if str(uni_val) + "_" + str(col) not in bins_represented_col:
                    bins_represented_col.append(str(uni_val) + "_" + str(col))

        bins_represented.append(bins_represented_col)
        interestigness_score += (len(bins_represented_col) / num_bins) / num_cols
    return interestigness_score, bins_represented


def jaccard_metric(summary, full_df=None, mapping_bin_values=None, bin_transform=False):
    if summary.shape[0] <= 0:
        print("the summary is empty")
        return 0.0

    mapping_bin_values['min'] = pd.to_numeric(
        mapping_bin_values['value'].apply(lambda x: str(x).replace('(', '').replace(')', '').split(',')[0]),
        downcast='float')
    mapping_bin_values['max'] = pd.to_numeric(
        mapping_bin_values['value'].apply(lambda x: str(x).replace('(', '').replace(']', '').split(',')[1]),
        downcast='float')
    if 'Unnamed: 0' in mapping_bin_values.columns:
        mapping_bin_values.drop('Unnamed: 0', axis=1, inplace=True)
    if bin_transform:
        summary = create_binned_summary(summary, mapping_bin_values)
    sum_jaccarad = 0

    for i in range(0, summary.shape[0]):
        for j in range(0, summary.shape[0]):
            sum_jaccarad += jaccard_score(np.array(summary.iloc[i]), np.array(summary.iloc[j]), average='micro')

    return sum_jaccarad / (summary.shape[0] ** 2)


def covariance_metric(summary, full_df):
    # dealing with nulls (in this case I filled with zero)
    # dealing with categorical numbers (I did basic encoding)
    summary_ent = summary.fillna(0).copy()
    full_df_ent = full_df[summary.columns].fillna(0).copy()
    for col in summary.columns:
        if summary[col].dtype == 'O' or hasattr(summary[col], 'cat'):  # category
            summary_ent[col] = summary[col].astype('category').cat.codes
            full_df_ent[col] = full_df[col].astype('category').cat.codes

    cov_mat_summary = np.cov(summary_ent.to_numpy().transpose())
    cov_mat_full = np.cov(full_df_ent.to_numpy().transpose())
    V = np.cov(np.array([cov_mat_summary.flatten(), cov_mat_full.flatten()]).T)
    IV = np.linalg.pinv(V)
    return distance.mahalanobis(cov_mat_summary.flatten(), cov_mat_full.flatten(), IV)


def eval_random(df, rules_final, mapping_bin_values, num_of_trials=100, num_cols=10, num_rows=10, must_have_column=None,
                metric_random_summarize='MAX', save_results=False):
    overall_score = defaultdict(float)
    best_summary = pd.DataFrame()
    for trial in range(0, num_of_trials):
        summary = generate_random_summary(df, num_cols=num_cols, num_rows=num_rows, must_have_column=must_have_column)
        summary = clean_summary(summary)
        dict_ev = evaluate_summary(summary, rules_final, mapping_bin_values, df)
        if trial == 0:
            overall_score = dict_ev
        for metric in dict_ev.keys():
            if metric_random_summarize == 'AVG':
                overall_score[metric] += dict_ev[metric][0]
            elif metric_random_summarize == 'MAX':
                if type(dict_ev[metric][0]) == float:
                    if dict_ev[metric][0] > overall_score[metric][0]:
                        overall_score[metric] = dict_ev[metric]
                        if metric == 'cell_cov':
                            best_summary = summary.copy()

        if trial % 2000 == 0:
            print("after {} tries, the max scores are:".format(trial))
            print(overall_score)
            if save_results:
                if len(save_results) > 0:
                    best_summary.to_csv(save_results)
                else:
                    best_summary.to_csv('best_summary_so_far.csv')

    for metric in dict_ev.keys():
        if metric_random_summarize == 'AVG':
            overall_score[metric] /= num_of_trials
        overall_score[metric] = [overall_score[metric]]

    return overall_score


# calculate cross entropy
def cross_entropy(p, q):
    return -sum([p[i] * log2(q[i]) for i in range(len(p))])


def cross_entropy_metric(summary, full_df):
    # dealing with data contains non-positive numbers (in this case I add the abs(min) number for each column)

    sum_ent = 0
    eps = 0.0001
    summary_ent = summary.fillna(0).copy()
    full_df_ent = full_df.copy()
    for col in summary.columns:
        if summary[col].dtype == 'O' or hasattr(summary[col], 'cat'):  # category
            summary_ent[col] = summary[col].astype('category').cat.codes
        if full_df[col].dtype == 'O' or hasattr(full_df[col], 'cat'):  # category
            full_df_ent[col] = full_df[col].astype('category').cat.codes

    for col in summary.columns:
        if summary_ent[col].min() <= 0:
            summary_ent[col] += abs(summary_ent[col].min()) + eps
    counter = 0
    for col_1 in summary.columns:
        for col_2 in summary.columns:
            sum_ent += cross_entropy(list(summary_ent[col_1]), list(summary_ent[col_2]))
            counter += 1
    return sum_ent / counter
