import math
import time
import pickle
import pandas as pd
import numpy as np



def group_datafiles_byID(meta_filepath, db_filepath):
    '''
        INPUT:
            meta_file: metadata file path. For example: '../datasets/raw/HT_Sensor_metadata.dat'
            db_file: main database file path. For example: '../datasets/raw/HT_Sensor_dataset.dat'

        OUTPUT:
            pandas dataframe
        
        DESCRIPTION:
            It joins both metadata and main database files using field 'id' as index. It also
            recalculates field 'time' according stimulus beginning.
    '''
    # Reading files
    df_meta = pd.read_csv(meta_filepath, delimiter='\t+', engine='python')
    df_db = pd.read_csv(db_filepath, delimiter='\s+', engine='python')

    # Setting new index
    df_db.set_index('id', inplace=True)
    # Joining both files
    df_db = df_db.join(df_meta, how='inner')
    # t0 indicates stimulus beginning
    df_db['time'] += df_db['t0']
    # Setting final index
    df_db.set_index(np.arange(df_db.shape[0]), inplace=True)

    return df_db



def reclassify_series_samples(df_db):
    '''
        INPUT:
            df_db: joint pandas dataframe.
        
        OUTPUT:
            pandas dataframe
        
        DESCRIPTION:
            It reclassifies as 'background' readings of series before stimulus and after stimulus.
    '''
    if 't0_delay' in df_db and 'dt_delay' in df_db:
        df_db.loc[df_db['time']<(df_db['t0']+df_db['t0_delay']), 'class'] = 'background'
        df_db.loc[df_db['time']>(df_db['t0']+df_db['dt']+df_db['dt_delay']), 'class'] = 'background'
    else:
        df_db.loc[df_db['time']<df_db['t0'], 'class'] = 'background'
        df_db.loc[df_db['time']>(df_db['t0']+df_db['dt']), 'class'] = 'background'

    return df_db



def split_series_byID(train_perc, joint_df):
    '''
        INPUT:
            n_ids: total number of IDs (in this project: 100)
            train_perc: percentage data of train set
            joint_df: joint pandas dataframe

        OUTPUT:
            df_train: pandas dataframe (train dataframe)
            df_test: pandas dataframe (test dataframe)

        DESCRIPTION:
            It chooses floor(train_perc * n_ids) indexes to train set, and the rest for test set,
            returning splitted dataframes.
    '''
    # Sampling test indices
    n_ids = len(set(joint_df['id']))
    n_train = math.floor(train_perc * n_ids)
    train_indices = np.random.choice(list(set(joint_df['id'])), size=n_train, replace=False)

    # Selecting train/test examples
    bool_list = []
    for id in joint_df.id:
        if id in train_indices:
            bool_list.append(True)
        else:
            bool_list.append(False)

    # We will use the fact it is a numpy array later
    bool_list = np.asarray(bool_list)

    df_train = joint_df[bool_list]
    # The fact that bool_list is a np array allows us to just choose its complementary
    df_test = joint_df[~bool_list]

    return df_train, df_test



def remove_excess_bg(df_db, delta=0.5):
    '''
        INPUT:
            df_db: pandas dataframe
            delta: real number. It determines what samples we want to 
                   keep before and after stimulus
        
        OUTPUT:
            pandas dataframe
        
        DESCRIPTION:
            It deletes excess background examples, returning modified dataframe.
    '''
    # Taking into account stimulus beginning/ending delay
    if 't0_delay' in df_db and 'dt_delay' in df_db:
        # condition 1 selectes all samples from stimulus beginning minus delta to the end
        cond1 = df_db['time'] > df_db['t0']+df_db['t0_delay']-delta
        # condition 2 selectes all samples from beginning to stimulus end plus delta
        cond2 = df_db['time'] < df_db['t0']+df_db['dt']+df_db['dt_delay']+delta

    else: 
        # condition 1 selectes all samples from stimulus beginning minus delta to the end
        cond1 = df_db['time'] > df_db['t0']-delta
        # condition 2 selectes all samples from beginning to stimulus end plus delta
        cond2 = df_db['time'] < df_db['t0']+df_db['dt']+delta
    
    # so we want to keep the samples that intersects both conditions
    final_cond = cond1 & cond2

    return df_db.loc[final_cond]




def _build_headers():
    return ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Temp.', 'Humidity', 'class', 'id', 't0', 'dt',
            't0_delay', 'dt_delay', 'R1_mean', 'R2_mean', 'R3_mean', 'R4_mean', 'R5_mean', 'R6_mean', 'R7_mean',
            'R8_mean', 'Temp._mean', 'Humidity_mean', 'R1_std', 'R2_std', 'R3_std', 'R4_std', 'R5_std', 'R6_std',
            'R7_std', 'R8_std', 'Temp._std', 'Humidity_std']


def window_df(df, window_size=120):
    '''
        INPUT:
            df: pandas dataframe
            window_size: positive integer. For safety, it is considered to be smaller than 5000
        
        OUTPUT:
            pandas dataframe (of size [df.shape[0] - (window_size*len(all_ids))]x[len(full_headers)] )
        
        DESCRIPTION:
            It calculates moving average for all provided sensors for given dataframe. It also computes
            standard deviation in the specified window.
    '''
    if window_size <= 0:
        raise ValueError('Introduced window size is too small: '+str(window_size))
    if window_size > 5000:
        raise ValueError('Introduced window size is too large: '+str(window_size))

    main_sensors = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Temp.', 'Humidity']
    main_head = main_sensors + ['class', 'id', 't0', 'dt', 't0_delay', 'dt_delay']
    full_headers = _build_headers()
    
    all_ids = list(set(df['id']))
    
    final_df_length = df.shape[0] - (window_size*len(all_ids))
    new_df = np.zeros((final_df_length, len(full_headers)), dtype=object) # + 1 dew to headers
    
    main_idx = 0

    for id in all_ids:
        # for the project, ID=95 is invalid
        if id != 95:
            # selecting sub-dataframe with corresponding ID
            dfid = df[df['id']==id]
            # init window pointer (selects window starts)
            window_ptr = 0

            while window_ptr+window_size < dfid.shape[0]:
                # calculating means and std devs
                vals = dfid[main_sensors].iloc[window_ptr : window_ptr+window_size]
                means = vals.mean()
                stds = vals.std()
                # selecting current sample sensor values
                new_data = dfid[main_head].iloc[window_ptr+window_size].to_numpy()
                # joining current sample data + window mean + window std dev
                new_data = np.hstack((new_data, means, stds))
                # appending new data to numpy matrix
                new_df[main_idx] = new_data
                # window moved
                window_ptr += 1
                main_idx += 1

    return pd.DataFrame(data=new_df, index=range(new_df.shape[0]), columns=full_headers)







if __name__ == '__main__':
    df_db = group_datafiles_byID('../datasets/raw/HT_Sensor_metadata.dat', '../datasets/raw/HT_Sensor_dataset.dat')
    print('=== Head before reclassify ===')
    print(df_db.head())

    print('=== Head after reclassify ===')
    df_db = reclassify_series_samples(df_db)
    print(df_db.head())

    df_db = remove_excess_bg(df_db)
    print(df_db.shape)

    '''
    store_filepath = '../datasets/preprocessed/joint_reclass.pkl'
    df_db.to_pickle(store_filepath)

    print('=== Head after writing and reading df ===')
    df_db = pd.read_pickle(store_filepath)
    print(df_db.head())
    '''