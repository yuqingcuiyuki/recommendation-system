import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
import numpy as np
import time


# read parquet file: train mat
train_mat = pd.read_parquet('Downloads/train_mat.parquet')

# display the first few rows of the dataframe
print(train_mat.head())
print(train_mat.shape)

print('reduce size of df by rating > 4')
train_mat=train_mat[train_mat['rating']!=1]
train_mat=train_mat[train_mat['rating']!=2]
train_mat=train_mat[train_mat['rating']!=3]
train_mat=train_mat[train_mat['rating']!=4]
#train_mat=train_mat[train_mat['rating']!=5]
#train_mat=train_mat[train_mat['rating']!=6]
#train_mat=train_mat[train_mat['rating']!=7]

print('remove recording_msid that listened <= 5 users')
msid_counts = train_mat['recording_msid'].value_counts()
valid_msid_values = msid_counts[msid_counts >=5].index
train_mat = train_mat[train_mat['recording_msid'].isin(valid_msid_values)]

print('reduced shape', train_mat.shape)

fractions=[0.25, 0.5, 0.75, 1]
num=100
i_alpha=1e-4
u_alpha=1e-4

for frac in fractions:
    # Randomly sample 25% of rows
    sample = train_mat.sample(frac=frac, random_state=42)

    # Map 'user_id' and 'recording_msid' to unique indices
    print('create mapping')
    user_id_mapping = {id: i for i, id in enumerate(sample['user_id'].unique())}
    recording_msid_mapping = {id: i for i, id in enumerate(sample['recording_msid'].unique())}

    print('start creating sparse matrix')
    matrix = coo_matrix((sample['rating'], 
                           (sample['user_id'].map(user_id_mapping), 
                            sample['recording_msid'].map(recording_msid_mapping))))
    print(matrix.shape)			
		  
    print('matrix done, start modeling')

    start_time = time.time()
    model = LightFM(loss='warp',
                    random_state=1004,
                    item_alpha=i_alpha, 
                    user_alpha=u_alpha, 
                    no_components=num)

    model.fit(matrix, epochs=10)

    print('model fit done')
    print('start prediciton')


    # get number of users and items
    n_users, n_items = matrix.shape

    # create an empty list to store the results
    results = []
    user_id_map=sample['user_id'].unique()
    recording_msid_map=sample['recording_msid'].unique()

    # iterate over all users
    for user_id in range(n_users):

        # get the scores for all items for this user
        scores = model.predict(user_id, np.arange(n_items))
        # print(scores)
        # get the top 100 item indices with the highest scores
        top_items = np.argsort(-scores)[:100]

        # append to the results
        results.append([user_id_map[user_id], [recording_msid_map[i] for i in list(top_items)]])

    print('prediction done')
    elapsed_time = time.time() - start_time
    print(f'Time taken for iteration with no_components={num}: {elapsed_time:.2f} seconds')



