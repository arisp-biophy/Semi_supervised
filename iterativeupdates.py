# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:05:23 2023

@author: AriSpiesberger
"""


import pandas as pd
import sklearn
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import scipy
from scipy.stats import binom,beta
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
def process_data(data):
    '''Processes data for statistical moment usage using sample stats function'''    
    ''' turns raw information for p2 responses into statistical moments'''
    
    # Initialize columns for storing posterior statistics
    columns = [ 'p2']
    stats = ['_means', '_samp_std', '_down', '_up']
    for col in columns:
        for stat in stats:
            data[col + stat] = 0

    # Iterate through each row in the DataFrame
    for i, row in enumerate(data.index):
        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print(i)
        
        # Calculate posterior statistics for each parameter (p1, p2, p3)
        for col in columns:
            data.loc[row, col + '_samp_std'], data.loc[row, col + '_means'], data.loc[row, col + '_down'], data.loc[row, col + '_up'] = sample_stats(data.loc[row, col + '_nsucc_x'], data.loc[row, col + '_nfail_x'], 5, 5)

    return data
def sample_stats(succ, fail, prior1, prior2):
    '''creates bayesian statistics from beta binomial method, uses succ and failures for binom and a = prior1, b = prior2'''
    
    # Update prior parameters with observed data (successes and failures)
    a = succ + prior1
    b = fail + prior2

    # Generate random samples from the posterior distribution
    dist = beta.rvs(a, b, size=1000)

    # Calculate the standard deviation of the posterior distribution
    conf = np.std(dist)

    # Calculate the 10th and 90th percentiles of the posterior distribution
    low = beta.ppf(0.1, a, b)
    up = beta.ppf(0.9, a, b)

    # Calculate the Maximum A Posteriori (MAP) estimate
    MAP = (succ + prior1) / (succ + fail + prior1 + prior2)

    return conf, MAP, low, up
def drop_duplicate_columns(df):
    '''occasionaly duplicate columns will occur in the concatenation of data, we should drop duplicates'''
    df = df.T.drop_duplicates().T
    return df

def replace_duplicates(df, column_name):
    '''replaces duplicate values of a column in pandas dataframe'''
    # get indices of duplicates
    duplicates = df[column_name].duplicated(keep=False)

    # replace values in duplicate rows with minimum value
    df.loc[duplicates, column_name] = df.loc[duplicates, column_name].groupby(df.loc[duplicates, column_name]).transform('min')

    # drop duplicate rows
    df = df.drop_duplicates(subset=column_name)

    return df

def drop_non_float_columns(df):
    '''drop all columns that arnt numerical'''
    return df.select_dtypes(include=[np.float64])
#read data
def drop_columns_without_prefix(df, prefix='p2'):
    '''clinical columns are labled with some phase, here we select that phase'''
    columns_to_drop = [col for col in df.columns if not (col.startswith(prefix) or col == 'calculated_end_date_x' or col == 'calculated_start_date_x')]
    df_filtered = df.drop(columns=columns_to_drop)
    return df_filtered

def relabel_duplicate_indices(df):
    counts = dict()
    new_indices = []

    for idx in df.index:
        if idx not in counts:
            counts[idx] = 0
            new_indices.append(idx)
        else:
            counts[idx] += 1
            new_idx = f"{idx}_{counts[idx]}"
            new_indices.append(new_idx)

    df.index = new_indices
    return df

def introduce_missing_labels(df, column, missing_percentage):
    '''unlables data from labled data'''
    # Calculate the number of labels to replace with missing values
    num_missing = int(len(df[column]) * missing_percentage / 100)

    # Randomly choose indices to replace with missing values
    missing_indices = np.random.choice(df.index, num_missing, replace=False)

    # Replace selected indices with missing values (2)
    df.loc[missing_indices, column] = 2

    return df


def load_data():
    #Read in all relevant data
    df1 = pd.read_csv('C:/Users/AriSpiesberger/Downloads/DavenAri/DavenAri/Raw_Ranks_CoreModel_V2.csv')
    df2 = pd.read_csv('C:/Users/AriSpiesberger/Downloads/Disease_Moments.csv.gz')
    df2 = df2[df2['disease_area'] == 'Type 2 Diabetes']
    df2 = df2[(df2['trial_phase'] == 'II') | (df2['trial_phase'] == 'I/II')]
    df3 = pd.read_csv('C:/Users/AriSpiesberger/Downloads/Target_Moments.csv.gz')
    df4 = pd.read_csv('C:/Users/AriSpiesberger/Downloads/MechanismOfAction_Moments.csv.gz')
    #configure unknowns for trial outcomes
    #often we have duplicate trial reports, we simply want the column minimum in this case
    df1 = replace_duplicates(df1,'trial_id')
    df2 = replace_duplicates(df2,'trial_id')
    df3 = replace_duplicates(df3,'trial_id')
    df4 = replace_duplicates(df4,'trial_id')

    #Join our datafreames for concatentation
    dfs = [df1, df2, df3,df4]

    #Now we merge these trials 1 by 1 over there trial ids
    df_merged = pd.merge(df1, df2, on='trial_id', how='inner')
    df_merged = pd.merge(df_merged, df3, on='trial_id', how='inner')
    df_merged = pd.merge(df_merged, df4, on='trial_id', how='inner')

    #We want to drop any duplicate data in the columns from the previous concatenation
    df_merged = drop_duplicate_columns(df_merged)

    #make the phases numerical
    dictionary = {'I':1, 'II':2,'III':3,'II/III':3,'I/II':2}
    df_merged = df_merged.replace({'trial_phase_x':dictionary})
    df_merged.index = df_merged['trial_id']
    df_merged = process_data(df_merged)
    
    return df_merged
def split_data(df_merged, train_size, random_state=42):
    # Separate known and unknown data
    X_known = df_merged[df_merged['trial_outcomes1_x'] != 2]
    X_unknown = df_merged[df_merged['trial_outcomes1_x'] == 2]

    # Split known data into train and test sets
    y_train = X_known['trial_outcomes1_x']
    X_train, X_test, y_train, y_test = train_test_split(X_known, y_train, test_size=1-train_size, random_state=random_state)

    # Drop the target column from test data
    X_test = X_test.drop(['trial_outcomes1_x', 'trial_outcomes1_y'], axis=1)

    # Drop unknown data target column
    X_unknown = X_unknown.drop(['trial_outcomes1_x', 'trial_outcomes1_y'], axis=1)

    # Get the pool data
    pool_data = X_unknown.drop('trial_id', axis=1)

    # Return the train, test, and pool data with their corresponding targets
    return X_train.drop('trial_outcomes1_x', axis=1), y_train, X_test, y_test, pool_data


df_merged = load_data()


X_known = df_merged[df_merged['trial_outcomes1_x'] != 2]
y_train = X_known['trial_outcomes1_x']
X_train, X_test, y_train, y_test = train_test_split(X_known, y_train, test_size=0.25)
#now we can introduce missing data to the 
df_merged = introduce_missing_labels(X_known, 'trial_outcomes1_x', 80)
X_known = df_merged[df_merged['trial_outcomes1_x'] != 2]
X_unknown = df_merged[df_merged['trial_outcomes1_x'] == 2]
'''
weights = {0: 0.7 , 1: 0.3}
class_counts = X_known['trial_outcomes1_x'].value_counts()
total_samples = len(X_known)
sample_sizes = {class_label: int(total_samples * weight) for class_label, weight in weights.items()}
X_known = pd.concat([X_known[X_known['trial_outcomes1_x'] == class_label].sample(n=sample_size, replace=True)
                          for class_label, sample_size in sample_sizes.items()])
'''

#now we relabel duplicate indices if they exist from the weighting assignments
X_known = relabel_duplicate_indices(X_known)
y_train = X_known['trial_outcomes1_x']
X_train = X_known.drop(['trial_outcomes1_x','trial_outcomes1_y'],axis = 1)
svc  = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=5)
#csvc  = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0)
pool_size = df_merged[df_merged['trial_outcomes1_x'] == 2].index
known = df_merged[df_merged['trial_outcomes1_x'] != 2].index

#Pool data if we are not assigning unlabeled data to the origional training set. unused otherwise
X_pool, y_pool = X_unknown.drop(['trial_outcomes1_x','trial_outcomes1_y'],axis = 1), X_unknown['trial_outcomes1_x']

# Remove the pool from the training data

X_train = drop_columns_without_prefix(X_train)
X_pool = drop_columns_without_prefix(X_pool)

X_test['trial_outcomes1_x'] = y_test
'''
weights = {0: 0.7 , 1: 0.3}
class_counts = X_test['trial_outcomes1_x'].value_counts()
total_samples = len(X_test)
sample_sizes = {class_label: int(total_samples * weight) for class_label, weight in weights.items()}
X_test = pd.concat([X_test[X_test['trial_outcomes1_x'] == class_label].sample(n=sample_size, replace=True)
                          for class_label, sample_size in sample_sizes.items()])
'''
y_test = X_test['trial_outcomes1_x']
X_test = X_test.drop('trial_outcomes1_x',axis = 1)
X_test = X_test[X_train.columns]
from scipy.stats import beta
import numpy as np

def updated_data_classification(data, targs):
    
    # Iterate through indexes and update the values
    for idx in data.index:
        start_date = data.loc[idx, 'calculated_start_date_x']
        print(idx)
        print(start_date)
        idts = data[data['calculated_end_date_x'] < start_date].index
        
        ans = targs.loc[idts]
        suc = len(ans[ans == 1])
        fail = len(ans[ans == 0])
        unk = len(ans[ans == 2])
        data.loc[idx, 'p2_nsucc_x'] = suc
        data.loc[idx, 'p2_nfail_x'] = fail
        data.loc[idx, 'p2_nunknown_x'] = unk

    return data
    
def self_training_classifier(X_train, y_train, X_pool, y_pool, threshold=0.98, max_iter=15):
    iteration = 0

    while iteration < max_iter and X_pool.shape[0] > 0:
        iteration += 1
        print(f"Iteration: {iteration}")

        # Train Gradient Boosting Classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1), y_train.astype('int'))

        # Make predictions on the unlabeled data
        probs = clf.predict_proba(X_pool.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1))

        # Select instances with prediction confidence above the threshold
        confident_indices = np.where(np.max(probs, axis=1) >= threshold)[0]
        predicted_labels = np.argmax(probs[confident_indices], axis=1)
        confident_pool_indices = X_pool.iloc[confident_indices].index

        if confident_indices.size == 0:
            print("No confident predictions found, stopping self-training.")
            break

        # Update pool labels with confident predictions
        y_pool.loc[confident_pool_indices] = predicted_labels

        # Combine training and pool data
        X_full = pd.concat([X_train, X_pool])
        y_full = pd.concat([y_train, y_pool])

        # Update and process the full data
        X_full = updated_data_classification(X_full, y_full)
        X_full = process_data(X_full)

        # Update training and pool data for the next iteration
        y_train = y_full[y_full != 2]
        y_pool = y_full[y_full == 2]
        X_train = X_full.loc[y_train.index]
        X_pool = X_full.loc[y_pool.index]

    return clf, X_full,y_full

#run the classifier to get x and y values from semi-supervised labeling
clf, X,y = self_training_classifier(X_train,y_train,X_pool,y_pool)
X = updated_data_classification(X,y)
X = process_data(X)
X['outcomes'] = y

#if we have data that still exists unlabled this will seperate it
X_semi = X[X['outcomes'] != 2]
y = X_semi['outcomes']
X_semi = X_semi.drop('outcomes',axis = 1)


#base and semi supervised classifier work in gradient boosted estimates
base = GradientBoostingClassifier(n_estimators = 120, learning_rate = 1.2, max_depth = 3, ccp_alpha = 0.00025) 
additive = GradientBoostingClassifier(n_estimators = 120, learning_rate = 1.2, max_depth = 3, ccp_alpha = 0.00025) 

#and fit data
base.fit(X_train.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1),y_train.astype('int'))
additive.fit(X_semi.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1),y.astype('int'))


y_pred_base = base.predict(X_test.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1))
y_pred_semi = additive.predict(X_test.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1))
base_prob = base.predict_proba(X_test.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1))
semi_prob = additive.predict_proba(X_test.drop(['calculated_end_date_x', 'calculated_start_date_x'], axis=1))

# Compute evaluation metrics
accuracy_base = accuracy_score(y_test.astype('int'), y_pred_base)
base_ent = log_loss(y_test.astype('int'), base_prob)
precision_base = precision_score(y_test.astype('int'), y_pred_base)
recall_base = recall_score(y_test.astype('int'), y_pred_base)
f1_base = f1_score(y_test.astype('int'), y_pred_base)

accuracy_semi = accuracy_score(y_test.astype('int'), y_pred_semi)
precision_semi = precision_score(y_test.astype('int'), y_pred_semi)
recall_semi = recall_score(y_test.astype('int'), y_pred_semi)
f1_semi = f1_score(y_test.astype('int'), y_pred_semi)
semi_ent = log_loss(y_test.astype('int'), semi_prob)

# Print summary statistics
print("Summary Statistics:")
print("\nClassifier 1 (Base):")
print(f"  Accuracy: {accuracy_base:.3f}")
print(f"  Precision: {precision_base:.3f}")
print(f"  Recall: {recall_base:.3f}")
print(f"  F1-score: {f1_base:.3f}")
print(f"  Cross-Entropy: {base_ent:3f}")

print("\nClassifier 2 (Semi-Supervised):")
print(f"  Accuracy: {accuracy_semi:.3f}")
print(f"  Precision: {precision_semi:.3f}")
print(f"  Recall: {recall_semi:.3f}")
print(f"  F1-score: {f1_semi:.3f}")
print(f"  Cross-Entropy: {semi_ent:3f}")    