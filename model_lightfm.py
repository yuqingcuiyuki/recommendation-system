import pandas as pd
from scipy.sparse import coo_matrix
from lightfm import LightFM
import numpy as np


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

# Map 'user_id' and 'recording_msid' to unique indices
print('create mapping')
user_id_mapping = {id: i for i, id in enumerate(train_mat['user_id'].unique())}
recording_msid_mapping = {id: i for i, id in enumerate(train_mat['recording_msid'].unique())}


print('start creating sparse matrix')
matrix = coo_matrix((train_mat['rating'], 
                           (train_mat['user_id'].map(user_id_mapping), 
                            train_mat['recording_msid'].map(recording_msid_mapping))))
print(matrix.shape)			
		  
print('matrix done, start modeling')

no_components=30 #10, 20, 30, later also 40, 50, 100
item_alpha=[1e-4,1e-2,1e-1]
user_alpha=[1e-4,1e-2,1e-1]

for i_alpha in item_alpha:
    for u_alpha in user_alpha:
	
        model = LightFM(loss='warp',
                        random_state=1004,
                        item_alpha=i_alpha, 
                        user_alpha=u_alpha, 
                        no_components=no_components)

        model.fit(matrix, epochs=10)

        print('model fit done')
        print('start prediciton')


        # get number of users and items
        n_users, n_items = matrix.shape

        # create an empty list to store the results
        results = []
        user_id_map=train_mat['user_id'].unique()
        recording_msid_map=train_mat['recording_msid'].unique()

        # iterate over all users
        for user_id in range(n_users):
            print('at user', user_id)
            # get the scores for all items for this user
            scores = model.predict(user_id, np.arange(n_items))
            # print(scores)
            # get the top 100 item indices with the highest scores
            top_items = np.argsort(-scores)[:100]
    
            # append to the results
            results.append([user_id_map[user_id], [recording_msid_map[i] for i in list(top_items)]])

        print('prediction done')
        print('store as df')
        # convert the results to a DataFrame
        result_df= pd.DataFrame(results, columns=['user_id', 'predictions'])

        print(result_df.head())


        print(f'save as pq for parameters: no_components={no_components}, item_alpha={i_alpha}, user_alpha={u_alpha}')
        result_df.to_parquet(f'lightfm_results/train_pred_{no_components}_{i_alpha}_{u_alpha}.parquet')		
        print('done') 


