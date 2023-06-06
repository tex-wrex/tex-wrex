import os
import pandas as pd

def acquire_motocycle_data():
    if os.path.exists('motocycle_data.csv'):
        df = pd.read_csv('motocycle_data.csv')
        return df
    else:
        url ='https://docs.google.com/spreadsheets/d/1jKNSokzayWWk-1_mcd-DikV9B_POOLHfuOCzV-AkI5Y/export?format=csv'
        df = pd.read_csv(url)
        url_2 = 'https://docs.google.com/spreadsheets/d/1dkvQy61WjfB769biDp7JQPXvrEHqPuofFUIlIWVo64I/export?format=csv'
        df_1 = pd.read_csv(url_2)
        merged_df = df.merge(df_1, left_on='crash_id', right_on='crash_id', how='left')
        merged_df.to_csv('motocycle_data.csv',index=False)

        return merged_df
    
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

def prepare_filtered_dataset_version():
    '''
    Takes the dataset from the 'acquire_motocycle_data()' function and filters out
    unnecessary data in order to replicate the dataset from Gabe's
    'master_modeling.csv' file.
    
    INPUT:
    NONE
    
    OUTPUT:
    master_modeling.csv = IF NONEXISTANT, .csv file of the filtered dataset
    new_df = Pandas dataframe of the filtered dataset
    '''
    if os.path.exists('master_modeling.csv'):
        new_df = pd.read_csv('master_modeling.csv', index_col=0)
        return new_df
    else:
        old_df = acquire_motocycle_data()
        remove_col_list = [col for col in old_df.columns.to_list() if col.endswith('_y')]
        keep_col_list = [col for col in old_df.columns.to_list() if col not in remove_col_list]
        new = old_df[keep_col_list]
        clean_col_list = [col.replace('_x', '') for col in keep_col_list]
        new.columns = clean_col_list
        new = new[~new.crash_id.isna()]
        filter_cols = [
        'person_age',
        'person_ethnicity',
        'person_gender',
        'has_motocycle_endorsment',
        'person_injury_severity',
        'vehicle_body_style',
        'vehicle_color',
        'vehicle_defect_1',
        'vehicle_make',
        'vehicle_model_name',
        'vehicle_model_year'
        ]
        filter_cols.insert(0, 'crash_id')
        new = new[filter_cols]
        for col in new:
            most_frequent_value = new[col].mode()[0]
            new[col].fillna(most_frequent_value, inplace=True)
        new.crash_id = new.crash_id.astype(int)
        new.person_age = new.person_age.astype(int)
        new_df = new
        new_df.to_csv('master_modeling.csv', index=False)
        return new_df

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def prepare_second_filtered_dataset_version():
    '''
    Takes the dataset from the 'acquire_motocycle_data()' function and filters out
    unnecessary data in order to replicate the dataset from Gabe's
    'master_modeling.csv' file.
    
    INPUT:
    NONE
    
    OUTPUT:
    master_modeling_updated.csv = IF NONEXISTANT, .csv file of the filtered dataset
    new_df = Pandas dataframe of the filtered dataset
    '''
    if os.path.exists('master_modeling_updated.csv'):
        new_df = pd.read_csv('master_modeling_updated.csv', index_col=0)
        return new_df
    else:
        df = prepare_filtered_dataset_version()
        df.drop('vehicle_defect_1', axis=1, inplace=True)
        vehicle_make_most_frequent = df['vehicle_make'].value_counts().idxmax()
        df['vehicle_make'] = df['vehicle_make'].replace('other', vehicle_make_most_frequent)
        df['vehicle_model_name'] = df['vehicle_model_name'].str.replace(r'\(.*\)', '', regex=True)
        df['vehicle_model_name'] = df['vehicle_model_name'].str.strip()
        unknown_model_indices = df[df['vehicle_model_name'] == 'unknown'].index
        for idx in unknown_model_indices:
            vehicle_make = df.loc[idx, 'vehicle_make']
            most_frequent_model = df[df['vehicle_make'] == vehicle_make]['vehicle_model_name'].value_counts().idxmax()
            df.loc[idx, 'vehicle_model_name'] = most_frequent_model
        df.rename(columns={'vehicle_model_name': 'vehicle_model'}, inplace=True)
        other_count = df[df['vehicle_model'].str.contains('other')].shape[0]
        other_model_indices = df[df['vehicle_model'].str.contains('other')].index
        for idx in other_model_indices:
            vehicle_make = df.loc[idx, 'vehicle_make']
            most_frequent_model = df[df['vehicle_make'] == vehicle_make]['vehicle_model'].value_counts().idxmax()
            df.loc[idx, 'vehicle_model'] = most_frequent_model
        df['vehicle_model_year'] = df['vehicle_model_year'].astype(str)
        df['vehicle_model_year'] = df['vehicle_model_year'].str.rstrip('.0')
        df.to_csv('master_modeling_updated.csv', index=False)
        df = pd.read_csv('master_modeling_updated.csv')
        vehicle_make_most_frequent = df['vehicle_make'].value_counts().idxmax()
        df['vehicle_make'] = df['vehicle_make'].replace(['other (explain in narrative)', 'unknown'], vehicle_make_most_frequent)
        changes_made = df['vehicle_make'].value_counts()[vehicle_make_most_frequent] - len(df[df['vehicle_make'] == vehicle_make_most_frequent])
        other_unknown_model_indices = df[df['vehicle_model'].str.contains('other|unknown')].index
        for idx in other_unknown_model_indices:
            vehicle_make = df.loc[idx, 'vehicle_make']
            most_frequent_model = df[df['vehicle_make'] == vehicle_make]['vehicle_model'].value_counts().idxmax()
            second_most_frequent_model = df[df['vehicle_make'] == vehicle_make]['vehicle_model'].value_counts().index[1] if len(df[df['vehicle_make'] == vehicle_make]['vehicle_model'].value_counts()) > 1 else most_frequent_model
            if 'other' in most_frequent_model or 'unknown' in most_frequent_model:
                df.loc[idx, 'vehicle_model'] = second_most_frequent_model
            else:
                df.loc[idx, 'vehicle_model'] = most_frequent_model
        changes_made = len(other_unknown_model_indices)
        new_df = df
        new_df.to_csv('master_modeling_updated.csv', index=False)
        return df