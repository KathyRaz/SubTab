import random
from SubTab import create_dict_patterns
from associations_rules_summary.utils_code import color_and_hover_summary

def generate_random_summary(df, num_cols=10, num_rows=10, must_have_column = None):
    chosen_columns_random = random.sample(list(set(df.columns)),k=num_cols)
    if must_have_column:
        if must_have_column not in chosen_columns_random:
            chosen_columns_random = chosen_columns_random[:-1]
            chosen_columns_random += [must_have_column]
    chosen_rows_random = random.sample(list(df.index),num_rows)
    summary = df.loc[chosen_rows_random]#.iloc[:,[chosen_columns_random]]
    summary = summary[list(chosen_columns_random)]#[chosen_columns_random]
    return summary

def color_random_summary(summary, rules, mapping_bins_to_values, df):
    dict_patterns = create_dict_patterns(summary=summary, rules=rules, mapping_bins_to_values=mapping_bins_to_values, df=df)
    df1 = color_and_hover_summary(summary[dict_patterns['order_columns']].round(2),
                                          dict_patterns['rule_dict'])
