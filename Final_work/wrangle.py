import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


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
        new_df = pd.read_csv('master_modeling.csv')
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
        new_df = pd.read_csv('master_modeling_updated.csv')
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
    
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
    
def prepare_third_filtered_dataset_version():
    '''
    Takes the dataset from the 'acquire_motocycle_data()' function and filters out
    unnecessary data in order to replicate the dataset from Gabe's
    'master_modeling_updated1.csv' file.
    
    INPUT:
    NONE
    
    OUTPUT:
    master_modeling_updated1.csv = IF NONEXISTANT, .csv file of the filtered dataset
    new_df = Pandas dataframe of the filtered dataset
    '''
    if os.path.exists('master_modeling_updated1.csv'):
        df = pd.read_csv('master_modeling_updated1.csv')
        return df
    else:
        df = prepare_second_filtered_dataset_version()
        make_country = {
            'honda': 'japan',
            'yamaha': 'japan',
            'suzuki': 'japan',
            'kawasaki': 'japan',
            'harley-davidson': 'usa',
            'bmw': 'Germany',
            'ducati': 'italy',
            'triumph': 'uk',
            'ktm': 'austria',
            'aprilia': 'italy',
            'indian': 'usa'
        }
        df['vehicle_make_country'] = df['vehicle_make'].map(make_country)
        df['vehicle_make_country'].fillna('Other', inplace=True)
        df['injury_binary'] = df['person_injury_severity'].apply(lambda x: 0 if x == 'n - not injured' else 1)
        df.to_csv('master_modeling_updated1.csv', index=False)
        return df
    
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
    
def split(df):
    '''
    This function splits a dataframe into 
    train, validate, and test in order to explore the data and to create and validate models. 
    It takes in a dataframe and contains an integer for setting a seed for replication. 
    Test is 20% of the original dataset. The remaining 80% of the dataset is 
    divided between valiidate and train, with validate being .30*.80= 24% of 
    the original dataset, and train being .70*.80= 56% of the original dataset. 
    The function returns, train, validate and test dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)   
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_age_visual():
    '''
    Displays the injury type distribution for age groups.

    INPUT:
    NONE

    OUTPUT:
    Visual
    '''
    master = prepare_third_filtered_dataset_version()
    bins = [0, 20, 30, 40, 50, 60, float('inf')]
    labels = ['<20', '20s', '30s', '40s', '50s', '60+']
    master['age_range'] = pd.cut(master['person_age'], bins=bins, labels=labels, right=False)
    cross_tab = pd.crosstab(master.age_range, master.person_injury_severity, normalize='index')
    desired_order = ['n - not injured', 'b - suspected minor injury', 'a - suspected serious injury', 'k - fatal injury']
    cross_tab = cross_tab[desired_order]
    colors = ['green', 'pink', 'darkorange', 'red']
    ax = cross_tab.plot(kind='bar', color=colors)
    plt.title('Injury Severity Percent Distribution By Age Group')
    plt.xlabel('Age Group')
    plt.xticks(rotation=0)
    plt.ylabel('Percentage')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    legend = plt.legend(loc='upper right', prop={'size': 9})
    legend.set_bbox_to_anchor((1.46, 1))
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_age_stattest():
    '''
    Displays the chi2_contingency statistic tests for all combinations of 
    injury type vs. age groups.

    INPUT:
    NONE

    OUTPUT:
    Statistic tests
    '''
    master = prepare_third_filtered_dataset_version()
    bins = [0, 20, 30, 40, 50, 60, float('inf')]
    labels = ['<20', '20s', '30s', '40s', '50s', '60+']
    master['age_range'] = pd.cut(master['person_age'], bins=bins, labels=labels, right=False)
    for vals in master.age_range.unique():
        for val in master.injury_binary.unique():
            observed = pd.crosstab(master.age_range == vals, master.injury_binary == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mAge Range:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mAge Range:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_ethnicity_visual():
    '''
    Displays the injury type distribution for ethnicity.

    INPUT:
    NONE

    OUTPUT:
    Visual
    '''
    master = prepare_third_filtered_dataset_version()
    cross_tab = pd.crosstab(master.person_ethnicity[~master.person_ethnicity.str.startswith(('98', '99'))], master.person_injury_severity, normalize='index')
    desired_order = ['n - not injured', 'b - suspected minor injury', 'a - suspected serious injury', 'k - fatal injury']
    cross_tab = cross_tab[desired_order]
    colors = ['green', 'pink', 'darkorange', 'red']
    ax = cross_tab.plot(kind='bar', color=colors)
    plt.title('Injury Severity Percent Distribution By Ethnicity')
    plt.xlabel('Ethnicity')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    legend = plt.legend(loc='upper right', prop={'size': 9})
    legend.set_bbox_to_anchor((1.46, 1))
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_ethnicity_stattest():
    '''
    Displays the chi2_contingency statistic tests for all combinations of 
    injury type vs. ethnicity.

    INPUT:
    NONE

    OUTPUT:
    Statistic tests
    '''
    master = prepare_third_filtered_dataset_version()
    for vals in master.person_ethnicity[~master.person_ethnicity.str.startswith(('98', '99'))].unique():
        for val in master.injury_binary.unique():
            observed = pd.crosstab(master.person_ethnicity == vals, master.injury_binary == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mEthnicity:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mEthnicity:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_endorsement_visual():
    '''
    Displays the injury type distribution for whether or not someone had a motorcycle endorsement.

    INPUT:
    NONE

    OUTPUT:
    Visual
    '''
    master = prepare_third_filtered_dataset_version()
    cross_tab = pd.crosstab(master.has_motocycle_endorsment, master.person_injury_severity, normalize='index')
    desired_order = ['n - not injured', 'b - suspected minor injury', 'a - suspected serious injury', 'k - fatal injury']
    cross_tab = cross_tab[desired_order]
    colors = ['green', 'pink', 'darkorange', 'red']
    ax = cross_tab.plot(kind='bar', color=colors)
    plt.title('Injury Severity Percent Distribution By Motorcycle Endorsement')
    plt.xlabel('Has Motorcycle Endorsement')
    plt.xticks(rotation=0, ticks=range(2), labels=['False', 'True'])
    plt.ylabel('Percentage')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    legend = plt.legend(loc='upper right', prop={'size': 9})
    legend.set_bbox_to_anchor((1.46, 1))
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_endorsement_stattest():
    '''
    Displays the chi2_contingency statistic tests for all combinations of 
    injury type vs. motorcycle endorsement.

    INPUT:
    NONE

    OUTPUT:
    Statistic tests
    '''
    master = prepare_third_filtered_dataset_version()
    for vals in master.has_motocycle_endorsment.unique():
        for val in master.injury_binary.unique():
            observed = pd.crosstab(master.has_motocycle_endorsment == vals, master.injury_binary == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mMotorcycle Endorsement:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mMotorcycle Endorsement:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_body_style_visual():
    '''
    Displays the injury type distribution for the body style of the motorcycle.

    INPUT:
    NONE

    OUTPUT:
    Visual
    '''
    master = prepare_third_filtered_dataset_version()
    cross_tab = pd.crosstab(master.vehicle_body_style, master.person_injury_severity, normalize='index')
    desired_order = ['n - not injured', 'b - suspected minor injury', 'a - suspected serious injury', 'k - fatal injury']
    cross_tab = cross_tab[desired_order]
    colors = ['green', 'pink', 'darkorange', 'red']
    ax = cross_tab.plot(kind='bar', color=colors)
    plt.title('Injury Severity Percent Distribution By Motorcycle Body Style')
    plt.xlabel('Motorcycle Body Style')
    plt.xticks(rotation=0, ticks=range(2), labels=['Motorcycle', 'Police Motorcycle'])
    plt.ylabel('Percentage')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    legend = plt.legend(loc='upper right', prop={'size': 9})
    legend.set_bbox_to_anchor((1.46, 1))
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_body_style_stattest():
    '''
    Displays the chi2_contingency statistic tests for all combinations of 
    injury type vs. motorcycle body style.

    INPUT:
    NONE

    OUTPUT:
    Statistic tests
    '''
    master = prepare_third_filtered_dataset_version()
    for vals in master.vehicle_body_style.unique():
        for val in master.injury_binary.unique():
            observed = pd.crosstab(master.vehicle_body_style == vals, master.injury_binary == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mMotorcycle Body Style:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mMotorcycle Body Style:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_color_visual():
    '''
    Displays the injury type distribution for the motorcycle color.

    INPUT:
    NONE

    OUTPUT:
    Visual
    '''
    master = prepare_third_filtered_dataset_version()
    cross_tab = pd.crosstab(master.vehicle_color.str.startswith(('bro', 'gld')), master.person_injury_severity, normalize='index')
    desired_order = ['n - not injured', 'b - suspected minor injury', 'a - suspected serious injury', 'k - fatal injury']
    cross_tab = cross_tab[desired_order]
    colors = ['green', 'pink', 'darkorange', 'red']
    ax = cross_tab.plot(kind='bar', color=colors)
    plt.title('Injury Severity Percent Distribution By Motorcycle Color (Brown, Gold)')
    plt.xlabel('Motorcycle Color')
    plt.xticks(rotation=0, ticks=range(2), labels=['Brown', 'Gold'])
    plt.ylabel('Percentage')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    legend = plt.legend(loc='upper right', prop={'size': 9})
    legend.set_bbox_to_anchor((1.46, 1))
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_color_stattest():
    '''
    Displays the chi2_contingency statistic tests for all combinations of 
    injury type vs. motorcycle color.

    INPUT:
    NONE

    OUTPUT:
    Statistic tests
    '''
    master = prepare_third_filtered_dataset_version()
    for vals in master.vehicle_color.unique():
        for val in master.injury_binary.unique():
            observed = pd.crosstab(master.vehicle_color == vals, master.injury_binary == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mMotorcycle Color:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mMotorcycle Color:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_make_country_visual():
    '''
    Displays the injury type distribution for motorcycle's make country.

    INPUT:
    NONE

    OUTPUT:
    Visual
    '''
    master = prepare_third_filtered_dataset_version()
    cross_tab = pd.crosstab(master.vehicle_make_country, master.person_injury_severity, normalize='index')
    desired_order = ['n - not injured', 'b - suspected minor injury', 'a - suspected serious injury', 'k - fatal injury']
    cross_tab = cross_tab[desired_order]
    colors = ['green', 'pink', 'darkorange', 'red']
    ax = cross_tab.plot(kind='bar', color=colors)
    plt.title('Injury Severity Percent Distribution By Motorcycle Make Country')
    plt.xlabel('Motorcycle Make Country')
    plt.xticks(rotation=0)
    plt.ylabel('Percentage')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    legend = plt.legend(loc='upper right', prop={'size': 9})
    legend.set_bbox_to_anchor((1.46, 1))
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
  
def get_motorcycle_make_country_stattest():
    '''
    Displays the chi2_contingency statistic tests for all combinations of 
    injury type vs. age groups.

    INPUT:
    NONE

    OUTPUT:
    Statistic tests
    '''
    master = prepare_third_filtered_dataset_version()
    for vals in master.vehicle_make_country.unique():
        for val in master.injury_binary.unique():
            observed = pd.crosstab(master.vehicle_make_country == vals, master.injury_binary == val)
            stat, p, dof, a = stats.chi2_contingency(observed)
            alpha = 0.05
            if p < 0.05:
                print(f'\033[32m========== REJECT NULL HYPOTHESIS ==========\033[0m\n\033[35mMake Country:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
            else:
                print(f'\033[31m========== ACCEPT NULL HYPOTHESIS ==========\033[0m\n\033[35mMake Country:\033[0m {vals}\n\033[35mInjury:\033[0m {val}\n\033[35mStatistic:\033[0m {stat}\n\033[35mP-Value:\033[0m {p}\n')
                
                
                
# Modeling
# =====================================================================================================================

def get_encoded_df():
    cols_to_drop = ['crash_id', 'person_injury_severity', 'injury_binary']
    df = prepare_third_filtered_dataset_version()
    df_encoded = df.copy()

    for col in df.columns:
        if col not in cols_to_drop:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True, dtype=int)
    return df_encoded


def get_safety_score():
    '''
    return the dataframe with safety_score. 
    
    '''
    
    if os.path.exists('modeling_with_safety_score.csv'):
        pd.read_csv('modeling_with_safety_score.csv')
    else:
        cols_to_drop = ['crash_id', 'person_injury_severity', 'injury_binary']
        df = prepare_third_filtered_dataset_version()
        df_encoded = df.copy()

        for col in df.columns:
            if col not in cols_to_drop:
                df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True, dtype=int)
        x = df_encoded.drop(columns=cols_to_drop)
        train, validate, test = split(df_encoded)
        X_train = train.drop(columns=cols_to_drop)
        y_train = train['injury_binary']

        X_validate = validate.drop(columns=cols_to_drop)
        y_validate = validate['injury_binary']

        X_test = test.drop(columns=cols_to_drop)
        y_test = test['injury_binary']
        pred_train = pd.DataFrame()
        pred_val = pd.DataFrame()
        pred_test = pd.DataFrame() 
        pred_val['actual'] = y_validate
        pred_test['actual'] = y_test
        # making the safety score 
        model = LogisticRegression(max_iter=1000)

        # Train the model
        model.fit(X_train, y_train)

        # Predict the probabilities of injury on the validation set
        y_val_pred = model.predict_proba(X_validate)[:, 1]


        pred_train['logistic'] = model.predict(X_train)
        pred_val['logistic'] = model.predict(X_validate)
        y_test_pred = model.predict_proba(X_test)[:, 1]
        df_encoded['safety_score'] = model.predict_proba(x)[:, 1]
        df['safety_score']= df_encoded.safety_score
        df.to_csv('modeling_with_safety_score.csv')

        return df
    
    
def get_modeling_metrics():
    cols_to_drop = ['crash_id', 'person_injury_severity', 'injury_binary']
    df = prepare_third_filtered_dataset_version()
    df_encoded = df.copy()

    for col in df.columns:
        if col not in cols_to_drop:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True, dtype=int)
    x = df_encoded.drop(columns=cols_to_drop)
    train, validate, test = split(df_encoded)
    X_train = train.drop(columns=cols_to_drop)
    y_train = train['injury_binary']

    X_validate = validate.drop(columns=cols_to_drop)
    y_validate = validate['injury_binary']

    X_test = test.drop(columns=cols_to_drop)
    y_test = test['injury_binary']


    # Lets make a new df to store our predictions so we can evaluate them later.
    pred_train = pd.DataFrame()
    pred_val = pd.DataFrame()
    pred_test = pd.DataFrame() 
    
    # set the datafrtames witht he actual values 
    pred_train['actual'] = y_train
    pred_val['actual'] = y_validate
    pred_test['actual'] = y_test
    
    # making the safety score 
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Predict the probabilities of injury on the validation set
    y_val_pred = model.predict_proba(X_validate)[:, 1]


    pred_train['logistic'] = model.predict(X_train)
    pred_val['logistic'] = model.predict(X_validate)
    y_test_pred = model.predict_proba(X_test)[:, 1]
    df_encoded['safety_score'] = model.predict_proba(x)[:, 1]
    df['safety_score']= df_encoded.safety_score
    
    pred_train['logistic'] = model.predict(X_train)
    # Descion Tree modeling
    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    
    pred_train['descion_tree'] = clf.predict(X_train)
    pred_val['descion_tree'] = clf.predict(X_validate)   
    pred_test['descion_tree'] = clf.predict(X_test)
    
    # random Forest modeling
    
    # make the model
    rf = RandomForestClassifier(max_depth= 10, random_state= 666)
    # fit the model
    rf.fit(X_train,y_train)
    # run the model
    pred_train['random_forest'] = rf.predict(X_train)
    # run the model to validate training model
    pred_val['random_forest'] = rf.predict(X_validate)
    
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=666).fit(X_train, y_train)
    pred_train['gbc'] = gbc.predict(X_train)
    pred_val['gbc'] = gbc.predict(X_validate)
    
    
    for cols in pred_train.columns:
        if not (pred_train[cols] == pred_train['actual']).all():
            print(f'classification_report for {cols} with training data')
            print('=====================================================')
            print(classification_report(pred_train.actual, pred_train[cols]))

    for cols in pred_val.columns:
        if not (pred_val[cols] == pred_val['actual']).all():
            print(f'classification_report for {cols} with validate data')
            print('=====================================================')
            print(classification_report(pred_val.actual, pred_val[cols]))

    for cols in pred_test.columns:
        if not (pred_test[cols] == pred_test['actual']).all():
            print(f'classification_report for {cols} with test data')
            print('=====================================================')
            print(classification_report(pred_test.actual, pred_test[cols]))
            
            

    


