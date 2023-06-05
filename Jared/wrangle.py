# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. acquire
4. prepare
5. wrangle
6. split
7. scale
8. sample_dataframe
9. remove_outliers_tukey
10. find_outliers_tukey
11. find_outliers_sigma
12. drop_nullpct
13. check_nulls
14. get_single_motorcycle_crashes
15. drop_nullpct_alternate
16. risk_scores_single_column
17. risk_scores_iterate_columns
18. gridsearchcv_hyperparameter_tuning
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for both the acquire & preparation phase of the data
science pipeline or also known as 'wrangling' the data...
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os

# =======================================================================================================
# Imports END
# Imports TO acquire
# acquire START
# =======================================================================================================

def acquire():
    '''
    Reads the 'master_list.csv' file acquired from all crashes with motorcycles in Texas from
    2018 - 2022 via CRIS QUERY (https://cris.dot.state.tx.us/public/Query/app/query-builder) data pull

    INPUT:
    NONE

    OUTPUT:
    master = pandas dataframe of master dataframe
    '''
    master = pd.read_csv('master_list.csv', index_col=0)
    return master

# =======================================================================================================
# acquire END
# acquire TO prepare
# prepare START
# =======================================================================================================

def prepare():
    '''
    Takes in the vanilla mass_shooters dataframe and returns a cleaned version that is ready 
    for exploration and further analysis

    INPUT:
    NONE

    OUTPUT:
    svcs = pandas dataframe of the prepared Texas crashes with only single motorcycle crashes
    '''
    if os.path.exists('svcs.csv'):
        svcs = pd.read_csv('svcs.csv', index_col=0)
        return svcs
    else:
        master = acquire()
        master.reset_index(inplace=True)
        master.columns = master.columns.str.replace(' ', '_').str.lower()
        master.fillna('No Data', inplace=True)
        master = master.replace(to_replace=re.compile(r'.*no\s*data.*', re.IGNORECASE), value='no data', regex=True)
        svcs = get_single_motorcycle_crashes(master)
        cols_to_remove = [
        'driver_drug_specimen_type',
        'person_drug_specimen_type',
        'person_drug_test_result',
        'driver_alcohol_result',
        'driver_alcohol_specimen_type',
        'person_alcohol_result',
        'person_alcohol_specimen_type_taken',
        'person_blood_alcohol_content_test_result',
        'crash_non-suspected_serious_injury_count',
        'crash_not_injured_count',
        'crash_possible_injury_count',
        'crash_suspected_serious_injury_count',
        'crash_total_injury_count',
        'crash_unknown_injury_count',
        'unit_non-suspected_serious_injury_count',
        'unit_not_injured_count',
        'unit_possible_injury_count',
        'unit_suspected_serious_injury_count',
        'unit_total_injury_count',
        'unit_unknown_injury_count',
        'person_non-suspected_serious_injury_count',
        'person_not_injured_count',
        'person_possible_injury_count',
        'person_suspected_serious_injury_count',
        'person_total_injury_count',
        'person_unknown_injury_count',
        'crash_death_count',
        'driver_time_of_death',
        'unit_death_count',
        'person_death_count',
        'person_time_of_death',
        'autonomous_level_engaged',
        'autonomous_unit_-_reported',
        'school_bus_flag',
        'bus_type',
        'cmv_actual_gross_weight',
        'cmv_cargo_body_type',
        'cmv_carrier_id_type',
        'cmv_disabling_damage_-_power_unit',
        'cmv_gvwr',
        'cmv_hazmat_release_flag',
        'cmv_intermodal_shipping_container_permit',
        'cmv_rgvw',
        'cmv_roadway_access',
        'cmv_sequence_of_events_1',
        'cmv_sequence_of_events_2',
        'cmv_sequence_of_events_3',
        'cmv_sequence_of_events_4',
        'cmv_total_number_of_axles',
        'cmv_total_number_of_tires',
        'cmv_trailer_disabling_damage',
        'cmv_trailer_gvwr',
        'cmv_trailer_rgvw',
        'cmv_trailer_type',
        'cmv_vehicle_operation',
        'cmv_vehicle_type',
        'vehicle_cmv_flag',
        'first_harmful_event',
        'first_harmful_event_involvement',
        'hazmat_class_1_id',
        'hazmat_class_2_id',
        'hazmat_id_number_1_id',
        'hazmat_id_number_2_id',
        'responder_struck_flag',
        'unit_description',
        'person_airbag_deployed',
        'person_ejected',
        'person_restraint_used',
        'highway_lane_design_for_hov,_railroads,_and_toll_roads',
        'railroad_company',
        'railroad_flag',
        'bridge_detail',
        'feature_crossed_by_bridge',
        'on_bridge_service_type',
        'under_bridge_service_type',
        'construction_zone_flag',
        'construction_zone_workers_present_flag',
        'commercial_motor_vehicle_flag',
        'date_arrived',
        'date_notified',
        'date_roadway_cleared',
        'date_scene_cleared',
        'direction_of_traffic'
        ]
        svcs.drop(columns=cols_to_remove, inplace=True)
        svcs, new_dict = drop_nullpct_alternate(svcs, 0.987)
        drop_negone_pct_dict = {
            'column_name' : [],
            'percent_nodata' : []
        }
        for col in svcs:
            pct = (svcs[col] == -1).sum() / svcs.shape[0]
            if pct > 0.987:
                svcs = svcs.drop(columns=col)
                drop_negone_pct_dict['column_name'].append(col)
                drop_negone_pct_dict['percent_nodata'].append(pct)
        svcs.person_injury_severity = svcs.person_injury_severity.str.replace('C - POSSIBLE INJURY', 'B - SUSPECTED MINOR INJURY')
        svcs = svcs[~(svcs.person_injury_severity == '99 - UNKNOWN')]
        times_list = svcs.crash_time.astype(str).to_list()
        fixed_times_list = []
        for val in times_list:
            if len(val) == 1:
                val = '000' + val
            elif len(val) == 2:
                val = '00' + val
            elif len(val) == 3:
                val = '0' + val
            fixed_times_list.append(val)
        svcs.crash_time = fixed_times_list
        svcs['crash_datetime'] = svcs.crash_date.str.strip() + ' ' + svcs.crash_time
        svcs.crash_datetime = pd.to_datetime(svcs.crash_datetime)
        svcs.reset_index(drop=True, inplace=True)
        return svcs

# =======================================================================================================
# prepare END
# prepare TO wrangle
# wrangle START
# =======================================================================================================

def wrangle():
    '''
    Function that acquires and prepares the Texas crash dataframe for use as well as creating a csv.

    INPUT:
    NONE

    OUTPUT:
    .csv = ONLY IF FILE NONEXISTANT
    svcs = pandas dataframe of the prepared Texas crashes with only single motorcycle crashes
    '''
    if os.path.exists('svcs.csv'):
        svcs = pd.read_csv('svcs.csv', index_col=0)
        svcs.crash_datetime = pd.to_datetime(svcs.crash_datetime)
        return svcs
    else:
        svcs = prepare()
        svcs.to_csv('svcs.csv', index_label=False)
        return svcs
    
# =======================================================================================================
# wrangle END
# wrangle TO split
# split START
# =======================================================================================================

def split(df, stratify=None):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets

    INPUT:
    df = pandas dataframe to be split into
    stratify = Splits data with specific columns in consideration

    OUTPUT:
    train = pandas dataframe with 70% of original dataframe
    validate = pandas dataframe with 20% of original dataframe
    test = pandas dataframe with 10% of original dataframe
    '''
    if stratify == None:
        train_val, test = train_test_split(df, train_size=0.9, random_state=1349)
        train, validate = train_test_split(train_val, train_size=0.778, random_state=1349)
    else:
        train_val, test = train_test_split(df, train_size=0.9, random_state=1349, stratify=df[stratify])
        train, validate = train_test_split(train_val, train_size=0.778, random_state=1349, stratify=train_val[stratify])
    return train, validate, test


# =======================================================================================================
# split END
# split TO scale
# scale START
# =======================================================================================================

def scale(train, validate, test, cols, scaler):
    '''
    Takes in a train, validate, test dataframe and returns the dataframes scaled with the scaler
    of your choice

    INPUT:
    train = pandas dataframe that is meant for training your machine learning model
    validate = pandas dataframe that is meant for validating your machine learning model
    test = pandas dataframe that is meant for testing your machine learning model
    cols = List of column names that you want to be scaled
    scaler = Scaler that you want to scale columns with like 'MinMaxScaler()', 'StandardScaler()', etc.

    OUTPUT:
    new_train = pandas dataframe of scaled version of inputted train dataframe
    new_validate = pandas dataframe of scaled version of inputted validate dataframe
    new_test = pandas dataframe of scaled version of inputted test dataframe
    '''
    original_train = train.copy()
    original_validate = validate.copy()
    original_test = test.copy()
    scale_cols = cols
    scaler = scaler
    scaler.fit(original_train[scale_cols])
    original_train[scale_cols] = scaler.transform(original_train[scale_cols])
    scaler.fit(original_validate[scale_cols])
    original_validate[scale_cols] = scaler.transform(original_validate[scale_cols])
    scaler.fit(original_test[scale_cols])
    original_test[scale_cols] = scaler.transform(original_test[scale_cols])
    new_train = original_train
    new_validate = original_validate
    new_test = original_test
    return new_train, new_validate, new_test

# =======================================================================================================
# scale END
# scale TO sample_dataframe
# sample_dataframe START
# =======================================================================================================

def sample_dataframe(train, validate, test):
    '''
    Takes train, validate, test dataframes and reduces the shape to no more than 1000 rows by taking
    the percentage of 1000/len(train) then applying that to train, validate, test dataframes.

    INPUT:
    train = Split dataframe for training
    validate = Split dataframe for validation
    test = Split dataframe for testing

    OUTPUT:
    train_sample = Reduced size of original split dataframe of no more than 1000 rows
    validate_sample = Reduced size of original split dataframe of no more than 1000 rows
    test_sample = Reduced size of original split dataframe of no more than 1000 rows
    '''
    ratio = 1000/len(train)
    train_samples = int(ratio * len(train))
    validate_samples = int(ratio * len(validate))
    test_samples = int(ratio * len(test))
    train_sample = train.sample(train_samples)
    validate_sample = validate.sample(validate_samples)
    test_sample = test.sample(test_samples)
    return train_sample, validate_sample, test_sample

# =======================================================================================================
# sample_dataframe END
# sample_dataframe TO remove_outliers_tukey
# remove_outliers_tukey START
# =======================================================================================================

def remove_outliers_tukey(df, col_list, k=1.5):
    '''
    Remove outliers from a dataframe based on a list of columns using the tukey method and then
    returns a single dataframe with the outliers removed

    INPUT:
    df = pandas dataframe
    col_list = List of columns that you want outliers removed
    k = Defines range for fences, default/normal is 1.5, 3 is more extreme outliers

    OUTPUT:
    df = pandas dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

# =======================================================================================================
# remove_outliers_tukey END
# remove_outliers_tukey TO find_outliers_tukey
# find_outliers_tukey START
# =======================================================================================================

def find_outliers_tukey(df, col_list, k=1.5):
    '''
    Find outliers from a dataframe based on a list of columns using the tukey method and then
    returns all of the values identifed as outliers

    INPUT:
    df = pandas dataframe
    col_list = List of columns that you want outliers removed
    k = Defines range for fences, default/normal is 1.5, 3 is more extreme outliers

    OUTPUT:
    NONE
    '''
    for col in col_list:
        lower_vals = []
        upper_vals = []
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - iqr * k
        upper_fence = q3 + iqr * k
        for val in df[col]:
            if val < lower_fence:
                lower_vals.append(val)
            elif val > upper_fence:
                upper_vals.append(val)
        print(f'\033[35m =========={col} // k={k}==========\033[0m')
        print(f'\033[32mValues < {lower_fence:.2f}: {len(lower_vals)} Values\033[0m')
        print(lower_vals)
        print(f'\n\033[32mValues > {upper_fence:.2f}: {len(upper_vals)} Values\033[0m')
        print(upper_vals)
        print(f'\n')

# =======================================================================================================
# find_outliers_tukey END
# find_outliers_tukey TO find_outliers_sigma
# find_outliers_sigma START
# =======================================================================================================

def find_outliers_sigma(df, col_list, sigma=2):
    '''
    Find outliers from a dataframe based on a list of columns using the three sigma rule and then
    returns all of the values identifed as outliers

    INPUT:
    df = pandas dataframe
    col_list = List of columns that you want outliers removed
    sigma = How many z-scores a value must at least be to identify as an outlier

    OUTPUT:
    NONE
    '''
    for col in col_list:
        mean = df[col].mean()
        std = df[col].std()
        z_scores = ((df[col] - mean) / std)
        outliers = df[col][z_scores.abs() >= sigma]
        print(f'\033[35m =========={col} // sigma={sigma}==========\033[0m')
        print(f'\033[32mMEAN:\033[0m {mean:.2f}')
        print(f'\033[32mSTD:\033[0m {std:.2f}')
        print(f'\033[32mOutliers:\033[0m {len(outliers)}')
        print(outliers)
        print(f'\n')

# =======================================================================================================
# find_outliers_sigma END
# find_outliers_sigma TO drop_nullpct
# drop_nullpct START
# =======================================================================================================

def drop_nullpct(df, percent_cutoff):
    '''
    Takes in a dataframe and a percent_cutoff of nulls to drop a column on
    and returns the new dataframe and a dictionary of dropped columns and their pct...
    
    INPUT:
    df = pandas dataframe
    percent_cutoff = Null percent cutoff amount
    
    OUTPUT:
    new_df = pandas dataframe with dropped columns
    drop_null_pct_dict = dict of column names dropped and pcts
    '''
    drop_null_pct_dict = {
        'column_name' : [],
        'percent_null' : []
    }
    for col in df:
        pct = df[col].isna().sum() / df.shape[0]
        if pct > percent_cutoff:
            df = df.drop(columns=col)
            drop_null_pct_dict['column_name'].append(col)
            drop_null_pct_dict['percent_null'].append(pct)
    new_df = df
    return new_df, drop_null_pct_dict

# =======================================================================================================
# drop_nullpct END
# drop_nullpct TO check_nulls
# check_nulls START
# =======================================================================================================

def check_nulls(df):
    '''
    Takes a dataframe and returns a list of columns that has at least one null value
    
    INPUT:
    df = pandas dataframe
    
    OUTPUT:
    has_nulls = List of column names with at least one null
    '''
    has_nulls = []
    for col in df:
        nulls = df[col].isna().sum()
        if nulls > 0:
            has_nulls.append(col)
    return has_nulls

# =======================================================================================================
# check_nulls END
# check_nulls TO get_single_motorcycle_crashes
# get_single_motorcycle_crashes START
# =======================================================================================================

def get_single_motorcycle_crashes(df):
    '''
    Takes in a pandas dataframe that needs to be filtered down to only single motorcycle crash incidents.
    This will only work if you have an existing 'crash_id' column.

    INPUT:
    df = Pandas dataframe with data that needs to be filtered down into only crashes with a single motorcycle

    OUTPUT:
    new_df = Filtered dataframe with only single motorcycle crashes
    '''
    original_df = df
    df = df[~df.crash_id.duplicated(keep=False)]
    df = df[df.person_type.str.startswith('5')]
    df = df[df.vehicle_body_style.str.startswith('MC') | df.vehicle_body_style.str.startswith('PM')]
    new_df = df
    return new_df

# =======================================================================================================
# get_single_motorcycle_crashes END
# get_single_motorcycle_crashes TO drop_nullpct_alternate
# drop_nullpct_alternate START
# =======================================================================================================

def drop_nullpct_alternate(df, percent_cutoff):
    '''
    THIS IS A MODIFIED VERSION THAT CHECKS FOR 'no data' INSTEAD OF NULLS!!!
    Takes in a dataframe and a percent_cutoff of 'no data' to drop a column on
    and returns the new dataframe and a dictionary of dropped columns and their pct...
    
    INPUT:
    df = pandas dataframe
    percent_cutoff = 'no data' percent cutoff amount
    
    OUTPUT:
    new_df = pandas dataframe with dropped columns
    drop_null_pct_dict = dict of column names dropped and pcts
    '''
    drop_nodata_pct_dict = {
        'column_name' : [],
        'percent_nodata' : []
    }
    for col in df:
        pct = (df[col] == 'no data').sum() / df.shape[0]
        if pct > percent_cutoff:
            df = df.drop(columns=col)
            drop_nodata_pct_dict['column_name'].append(col)
            drop_nodata_pct_dict['percent_nodata'].append(pct)
    new_df = df
    return new_df, drop_nodata_pct_dict

# =======================================================================================================
# drop_nullpct_alternate END
# drop_nullpct_alternate TO risk_scores_single_column
# risk_scores_single_column START
# =======================================================================================================

def risk_scores_single_column(df, col_name):
    '''
    Takes in a pandas dataframe and the specific column that you want a specific
    risk value for each specific value in the column...
    Gives a list of risk values associated by each value in the column as well
    as the dictionary of unique values and their risk value...
    
    INPUT:
    df = Pandas dataframe for utilization
    col_name = The exact column name within 'df' that you want
    
    OUTPUT:
    risk_value_list = List of risk values by each value in the column
    '''
    risk_dict = {
    'Unique Value' : [],
    'Risk Score' : []
    }
    for unique_value in df[col_name].unique():
        injury_distribution_dict = df[df[col_name] == unique_value]['person_injury_severity'].value_counts(normalize=True).to_dict()
        for injury_type in injury_distribution_dict.keys():
            if injury_type == 'A - SUSPECTED SERIOUS INJURY':
                injury_distribution_dict['A - SUSPECTED SERIOUS INJURY'] = injury_distribution_dict['A - SUSPECTED SERIOUS INJURY'] * 2
            elif injury_type == 'K - FATAL INJURY':
                injury_distribution_dict['K - FATAL INJURY'] = injury_distribution_dict['K - FATAL INJURY'] * 3
        risk_score = sum(injury_distribution_dict.values()) / 3
        risk_dict['Unique Value'].append(unique_value)
        risk_dict['Risk Score'].append(risk_score)
    risk_value_list = []
    for value in df[col_name]:
        risk_index = risk_dict['Unique Value'].index(value)
        risk = risk_dict['Risk Score'][risk_index]
        risk_value_list.append(risk)
    return risk_dict, risk_value_list

# =======================================================================================================
# risk_scores_single_column END
# risk_scores_single_column TO risk_scores_iterate_columns
# risk_scores_iterate_columns START
# =======================================================================================================

def risk_scores_iterate_columns(df, col_list):
    '''
    Takes in a pandas dataframe and a list of column names that you want to get a risk
    score from and get an overal aggregate risk score by row...
    Gives a list of aggregated risk values associated by each row in the dataframe as well
    as a list of dictionaries for risk scores in each column...
    
    INPUT:
    df = Pandas dataframe for utilization
    col_list = List of exact column names you want an aggregate risk score from
    
    OUTPUT:
    agg_risk_value_list = List of aggregated risk values by each value in the column
    column_risk_value_list_of_dicts = List of dictionaries for each column and their risk scores
    '''
    col_and_risk_dict = {}
    for col in col_list:
        risk_score_dict, risk_score_list = risk_scores_single_column(df, col)
        col_and_risk_dict[col] = risk_score_list
    row_sum = pd.DataFrame(col_and_risk_dict).sum(axis=1).to_list()
    column_agg = []
    for value in row_sum:
        row_agg = value / len(col_and_risk_dict)
        column_agg.append(row_agg)
    agg_risk_value_list = column_agg
    column_risk_value_list_of_dicts = col_and_risk_dict
    return agg_risk_value_list, column_risk_value_list_of_dicts

# =======================================================================================================
# risk_scores_iterate_columns END
# risk_scores_iterate_columns TO gridsearchcv_hyperparameter_tuning
# gridsearchcv_hyperparameter_tuning START
# =======================================================================================================

def gridsearchcv_hyperparameter_tuning(train_df, validate_df, feature_selection_list, target_feature):
    '''
    Takes in a train and validate pandas datasets, list of column names to train models
    off of, and the target column name in order to find the best hyperparameters. 
    For classification models like DecisionTreeClassifier, RandomForestClassifier,
    K-Nearest Neighbors, and Logistic Regression and returns a pandas dataframe
    with the best perfroming model for each unique classification model.
    
    IMPORTANT:
    This currently DOES NOT encode anything, so please ensure features are
    encoded prior to input.
    
    INPUT:
    train_df = Pandas dataframe of the training dataset
    validate_df = Pandas dataframe of the validation dataset
    feature_selection_list = List of column names to train model off of
    target_feature = Column name of the target variable
    
    OUTPUT:
    df_model_results = Pandas dataframe of the best performing model per
                       unique classification model
    '''
    train_x = train_df[feature_selection_list]
    train_y = train_df[target_feature]
    validate_x = validate_df[feature_selection_list]
    validate_y = validate_df[target_feature]
    models = {
    'Decision Tree' : (DecisionTreeClassifier(), {'max_depth' : [None] + list(range(3, 15)),
                                                  'min_samples_split' : list(range(2, 4)),
                                                  'min_samples_leaf' : list(range(1, 3)),
                                                  'random_state' : [1776]}),
    'Random Forest' : (RandomForestClassifier(), {'n_estimators' : [100, 200, 300],
                                                  'max_depth' : [None] + list(range(3, 15)),
                                                  'min_samples_split' : list(range(2, 4)),
                                                  'min_samples_leaf' : list(range(1, 3)),
                                                  'random_state' : [1776]}),
    'KNN' : (KNeighborsClassifier(), {'n_neighbors' : [5, 10, 50],
                                      'weights' : ['uniform', 'distance'],
                                      'algorithm' : ['ball_tree', 'kd_tree', 'brute', 'auto']}),
    'Logistic Regression' : (LogisticRegression(), {'C' : [0.1, 1, 10],
                                                    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                                                    'random_state' : [1776]})
    }
    model_results = []
    for model_name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
        grid_search.fit(train_x, train_y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        train_accuracy = grid_search.best_score_
        validate_accuracy = best_model.score(validate_x, validate_y)
        model_results.append({
            'Model': model_name,
            'Best Estimator': best_model,
            'Best Parameters': best_params,
            'Train Accuracy': train_accuracy,
            'Validate Accuracy': validate_accuracy
        })
    df_model_results = pd.DataFrame(model_results)
    return df_model_results

# =======================================================================================================
# gridsearchcv_hyperparameter_tuning END
# gridsearchcv_hyperparameter_tuning TO 
#  START
# =======================================================================================================