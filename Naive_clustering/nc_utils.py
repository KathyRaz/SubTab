from sklearn.preprocessing import LabelEncoder
import uuid
from word2vec_embedding.utils_w2v import data_transformation
from sklearn.cluster import KMeans
import numpy as np
import random
def gen_model_uuid(label=None):
    uu = uuid.uuid4().hex
    # os.path.join(
    # os.mkdir(uu)
    if label:
        return f"{uu}_{label}"
    else:
        return uu
    
    
def gen_nc_summary(df, n_rows, n_cols,clustering_algo='KMeans'):
    name_dataset = 'dataset'
    MODEL_UUID = None
    MODEL_DESCRIPTION = "{}_binned_data_cell_to_vec".format(name_dataset)
    if not MODEL_UUID:
        MODEL_UUID = gen_model_uuid(MODEL_DESCRIPTION)
    prefix = ""
#     dataset, full_df, mapping_bins_to_values = data_transformation(MODEL_UUID=MODEL_UUID, name_dataset=name_dataset,
#                                                                    prefix=prefix,
#                                                                    df=df.fillna(0).copy(), q=10, precision=0,
#                                                                    path_df_binned=None,
#                                                                    trim_multi_val_col=False, create_mapping=True,
#                                                                    path_mapping=None,
#                                                                    config_params=None, save_data=True)
    # creating instance of labelencoder
    
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    thershold = 100000
    if df.shape[0]>100000:
        df = df.sample(n=thershold, random_state=1)
    df = df.fillna('null')
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = labelencoder.fit_transform(df[c])
            except:
                df[c] = labelencoder.fit_transform(df[c].astype(str))
                
        elif df[c].dtype == 'category':
            df[c] =df[c].cat.codes
    vec_list_rows = df.values

    if clustering_algo == 'KMeans':
        model_rows = KMeans(n_clusters=n_rows, random_state=0).fit(vec_list_rows)
    centroids_rows = model_rows.cluster_centers_

    summary_idx_rows = []
    for c in centroids_rows:
        idx = np.argmin(np.linalg.norm(c - vec_list_rows, axis=1))
        summary_idx_rows.append(idx)

    summary_wo_bins_all_columns = df.iloc[summary_idx_rows].copy()

    # cols
    if n_cols >= summary_wo_bins_all_columns.shape[1]:
        return summary_wo_bins_all_columns

        
    vec_list_cols = df.T.values
    if clustering_algo == 'KMeans':
        model_cols = KMeans(n_clusters=n_cols, random_state=0).fit(vec_list_cols)
    centroids_cols = model_cols.cluster_centers_
    summary_idx_cols = []
    for c in centroids_cols:
        idx = np.argmin(np.linalg.norm(c - vec_list_cols, axis=1))
        summary_idx_cols.append(idx)

    summary = summary_wo_bins_all_columns.iloc[:, summary_idx_cols]
    
    return summary