# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. acquire
4. prepare
5. wrangle
6. visual1
7. visual2
8. visual3
9. visual4
10. stat1
11. stat2
12. stat3
13. stat4
14. models
15. topmodels
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions in order to expedite and maintain cleanliness of
the work presented within the 'final_report.ipynb'
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import re
import wrangle as w
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
import os

# Set default matplotlib plot style to 'bmh'
mpl.style.use('bmh')

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
        svcs = w.get_single_motorcycle_crashes(master)
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
        svcs, new_dict = w.drop_nullpct_alternate(svcs, 0.987)
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
# wrangle TO visual1
# visual1 START
# =======================================================================================================

def visual1():
    '''
    Returns the first specific visual for the 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Visualizations
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        hour_counts = train[train.person_injury_severity == val].groupby(by=train.crash_datetime.dt.hour)['person_injury_severity'].count()
        sns.barplot(x=hour_counts.index, y=hour_counts, color='darkblue')
        plt.title(f'Distribution of {val} vs. Time of Crash')
        plt.xlabel('Hour of Crash (24-Hour)')
        plt.ylabel('Count')
        plt.show()

# =======================================================================================================
# visual1 END
# visual1 TO visual2
# visual2 START
# =======================================================================================================

def visual2():
    '''
    Returns the second specific visual for the 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Visualizations
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        dayofweek_counts = train[train.person_injury_severity == val].groupby(by=train.crash_datetime.dt.dayofweek)['person_injury_severity'].count()
        sns.barplot(x=dayofweek_counts.index, y=dayofweek_counts, color='darkblue')
        plt.title(f'Distribution of {val} vs. Crash Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Count')
        plt.show()

# =======================================================================================================
# visual2 END
# visual2 TO visual3
# visual3 START
# =======================================================================================================

def visual3():
    '''
    Returns the third specific visual for the 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Visualizations
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        weather_counts = train[train.person_injury_severity == val].groupby('weather_condition')['person_injury_severity'].count()
        sns.barplot(x=weather_counts.index, y=weather_counts, color='darkblue')
        plt.title(f'Distribution of {val} vs. Weather Condition')
        plt.xlabel('Weather Condition')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.show()

# =======================================================================================================
# visual3 END
# visual3 TO visual4
# visual4 START
# =======================================================================================================

def visual4():
    '''
    Returns the fourth specific visual for the 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Visualizations
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        intersections_counts = train[train.person_injury_severity == val].groupby('intersection_related')['person_injury_severity'].count()
        sns.barplot(x=intersections_counts.index, y=intersections_counts, color='darkblue')
        plt.title(f'Distribution of {val} vs. Intersection Relation')
        plt.xlabel('Intersection Relation to Crash')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.show()

# =======================================================================================================
# visual4 END
# visual4 TO stat1
# stat1 START
# =======================================================================================================

def stat1():
    '''
    Returns the statistical test(s) for visual1 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Statistical Test(s)
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        observed = pd.crosstab(train.crash_datetime.dt.hour, train.person_injury_severity == val)
        stat, p, dof, a = stats.chi2_contingency(observed)
        alpha = 0.05
        if p < 0.05:
            print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35m{val}\nStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
        else:
            print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35m{val}\nStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# =======================================================================================================
# stat1 END
# stat1 TO stat2
# stat2 START
# =======================================================================================================

def stat2():
    '''
    Returns the statistical test(s) for visual2 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Statistical Test(s)
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        for vals in train.crash_datetime.dt.dayofweek.unique():
            observed = pd.crosstab(train.crash_datetime.dt.dayofweek, train.person_injury_severity == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mCondition:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mCondition:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# =======================================================================================================
# stat2 END
# stat2 TO stat3
# stat3 START
# =======================================================================================================

def stat3():
    '''
    Returns the statistical test(s) for visual3 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Statistical Test(s)
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        for vals in train.weather_condition.unique():
            observed = pd.crosstab(train.weather_condition == vals, train.person_injury_severity == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mCondition:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mCondition:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# =======================================================================================================
# stat3 END
# stat3 TO stat4
# stat4 START
# =======================================================================================================

def stat4():
    '''
    Returns the statistical test(s) for visual4 'final_report.ipynb'...

    INPUT:
    NONE

    OUTPUT:
    Statistical Test(s)
    '''
    master = w.wrangle()
    crash_conditions_cols= [
    'crash_id',
    'person_injury_severity',
    'crash_date',
    'crash_datetime',
    'crash_month',
    'crash_time',
    'crash_year',
    'day_of_week',
    'weather_condition',
    'light_condition',
    'surface_condition',
    'contributing_factor_1',
    'contributing_factor_2',
    'contributing_factor_3',
    'possible_contributing_factor_1',
    'possible_contributing_factor_2',
    'other_factor',
    'intersection_related',
    'manner_of_collision',
    'object_struck',
    'roadway_alignment',
    'roadway_relation',
    'roadway_type',
    'speed_limit'
    ]
    crash_df = master[crash_conditions_cols]
    train, validate, test = w.split(crash_df, stratify='person_injury_severity')
    for val in train.person_injury_severity.unique():
        for vals in train.intersection_related.unique():
            observed = pd.crosstab(train.intersection_related == vals, train.person_injury_severity == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mCondition:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mCondition:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# =======================================================================================================
# stat4 END
# stat4 TO models
# models START
# =======================================================================================================

def models():
    '''
    Shows the best model created from an iteration of gridsearchcv for each unique classification
    model and returns it as a pandas dataframe.

    INPUT:
    NONE

    OUTPUT:
    models_df = Pandas dataframe of best model for each unique classification model
    '''
    models_df = {'Model': {0: 'Decision Tree',
                           1: 'Random Forest',
                           2: 'KNN',
                           3: 'Logistic Regression',
                           4: 'Baseline'},
                'Best Estimator': {0: DecisionTreeClassifier(max_depth=3, random_state=1776),
                                   1: RandomForestClassifier(max_depth=3, random_state=1776),
                                   2: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=50),
                                   3: LogisticRegression(C=10, random_state=1776, solver='sag'),
                                   4: Baseline(mode)},
                'Best Parameters': {0: {'max_depth': 3,
                                        'min_samples_leaf': 1,
                                        'min_samples_split': 2,
                                        'random_state': 1776},
                                    1: {'max_depth': 3,
                                        'min_samples_leaf': 1,
                                        'min_samples_split': 2,
                                        'n_estimators': 100,
                                        'random_state': 1776},
                                    2: {'algorithm': 'ball_tree', 'n_neighbors': 50, 'weights': 'uniform'},
                                    3: {'C': 10, 'random_state': 1776, 'solver': 'sag'},
                                    4: {'Baseline-Mode'}},
                'Train Accuracy': {0: 0.6117491569838304,
                                   1: 0.6123532847160152,
                                   2: 0.6088263792963353,
                                   3: 0.6039898533355001,
                                   4: 0.575}}
    models_df = pd.DataFrame(models_df)
    return models_df

# =======================================================================================================
# models END
# models TO topmodels
# topmodels START
# =======================================================================================================



# =======================================================================================================
# topmodels END
# topmodels TO 
#  START
# =======================================================================================================