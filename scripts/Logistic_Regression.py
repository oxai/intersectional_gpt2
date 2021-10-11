# -*- coding: utf-8 -*-
"""Logistic_Regression.py


"""# Imports and Installs"""

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from scipy import stats
from ast import literal_eval
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import math

"""# Functions"""

## Thresholding and Aggregation
def make_count_df(freq_matrix, gender, intersection, CUM_THRESHOLD):
  '''Return counts and ranks of job distribution'''
  # Subset dataframe by intersection
  if intersection == 'continent':
    subset_df = freq_matrix.drop('name', axis = 1)
  elif intersection == 'average':
    subset_df = freq_matrix.drop('name', axis = 1)
  else:
    if intersection == 'religion':
      subs = ['buddhist','christian', 'hindu', 'muslim', 'jewish']
    elif intersection == 'ethnicity':
      subs = ['asian', 'black', 'hispanic', 'white']
    elif intersection == 'sexuality':
      subs = ['gay', 'straight']
    elif intersection == 'political':
      subs = ['conservative', 'liberal']
    elif intersection == 'base':
        subs = ['base'] 
    subset_df = freq_matrix[freq_matrix['category'].isin(subs)]
    subset_df = subset_df.drop('name', axis = 1)
  if gender != 'BOTH':
      subset_df = subset_df[subset_df['gender'] == gender]
  # Calculate counts and proportions per job
  subset_df = subset_df.set_index(['category', 'gender'])
  subset_df = subset_df.sum(axis = 0)
  subset_df = subset_df.reset_index()
  subset_df = subset_df.rename(columns = {'index': 'job', 0:'count'})
  subset_df = subset_df.sort_values(by = 'count', ascending = False)
  subset_df['share'] = subset_df['count']/subset_df['count'].sum()
  subset_df.index = pd.RangeIndex(start=1, stop=len(subset_df)+1, step=1)
  subset_df = subset_df.reset_index().rename(columns = {'index': 'rank'})
  # Keep cumulative > CUM_THRESHOLD:
  subset_df['cum_total'] = subset_df['share'].cumsum()
  cum_total = subset_df[subset_df['cum_total'] >= CUM_THRESHOLD]
  cum_total.index = pd.RangeIndex(start=0, stop=len(cum_total), step=1)
  xline = cum_total.iloc[0]['rank']
  yline = cum_total.iloc[0]['share']
  bar_points = [xline, yline]
  return subset_df, bar_points

def threshold_check(freq_matrix, intersection, THRESHOLD):
  ''' Calculates cumulative proportion of jobs < THRESHOLD '''
  # Subset dataframe by intersection
  if intersection == 'continent':
    subset_df = freq_matrix
    subs = ['None']
  else:
    if intersection == 'religion':
      subs = ['christian', 'buddhist', 'hindu', 'jewish' ,'muslim']
    elif intersection == 'ethnicity':
      subs = ['asian', 'black', 'hispanic', 'white']
    elif intersection == 'sexuality':
      subs = ['gay', 'straight']
    elif intersection == 'political':
      subs = ['conservative', 'liberal']
    elif intersection == 'base':
        subs = ['base'] 
    subset_df = freq_matrix[freq_matrix['category'].isin(subs)]  
  df_frequency = subset_df.groupby(['category', 'gender']).aggregate(sum).astype(int)
  thresholds = df_frequency.copy()
  df_frequency = subset_df.groupby(['category','gender']).aggregate(sum).astype(int)
  thresholds = df_frequency.copy()
  # Remove jobs < THRESHOLD
  for token in df_frequency.columns:
    if df_frequency[token].aggregate(sum) < THRESHOLD:
      thresholds = thresholds.drop(columns = token)
  return thresholds, subs

def threshold_tables(freq_matrix_idents, freq_matrix_names):
  ''' Prints threshold, total tokens and cumulative proportion < THRESHOLD'''
  intersections = ['base', 'ethnicity', 'religion', 'sexuality', 'political', 'continent']
  total_calls = [14000, 56000, 70000, 28000, 28000, 200000]
  thresholds = [0.0025*x for x in total_calls]
  threshold_dict = {}
  for intersection, THRESHOLD in zip(intersections, thresholds):
    if intersection == 'continent':
      thresholds= threshold_check(freq_matrix_names, intersection, THRESHOLD)[0]
      total_sum = thresholds.sum().sum()
      total_tokens = len(freq_matrix_names)
      print(total_tokens)
    else:
      thresholds, subs = threshold_check(freq_matrix_idents, intersection, THRESHOLD)
      total_sum = thresholds.sum().sum()
      total_tokens = len(freq_matrix_idents[freq_matrix_idents['category'].isin(subs)])
      total_tokens = len(freq_matrix_idents[freq_matrix_idents['category'].isin(subs)])                   
    print(f'{intersection}, threshold = {THRESHOLD}, total_tokens = {total_tokens}: proportion of counts {np.round(total_sum/total_tokens, 3)}')
    threshold_dict[intersection] = THRESHOLD
  
  return threshold_dict


## Regression
def make_regression_df(_freq_matrix_IDENTITIES, _freq_matrix_NAMES, intersection, criteria, THRESHOLD):
  '''Function to split dataframe by intersection, with base category always included so it can be used in the regression'''
  # Subset dataframe
  if intersection == 'continent':
    subset_df = _freq_matrix_NAMES
  else:
    if intersection == 'religion':
      subs = ['base', 'christian', 'buddhist', 'hindu', 'jewish', 'muslim', 'sikh']
    elif intersection == 'ethnicity':
      subs = ['base', 'asian', 'black', 'hispanic', 'white']
    elif intersection == 'sexuality':
      subs = ['base', 'gay', 'straight']
    elif intersection == 'political':
      subs = ['base', 'conservative', 'liberal']
    elif intersection == 'base':
      subs = ['base']
    subset_df = _freq_matrix_IDENTITIES[_freq_matrix_IDENTITIES['category'].isin(subs)]
  df_frequency = subset_df.groupby(['category', 'gender']).aggregate(sum).astype(int)
  thresholds = df_frequency.copy()
  # Apply threshold
  for token in df_frequency.columns:
    if df_frequency[token].aggregate(sum) < THRESHOLD:
      thresholds = thresholds.drop(columns = token)
  proportions = thresholds.div(thresholds.aggregate(sum))
  # Selection criteria
  if criteria == 'range':
    selection = proportions.aggregate(max) - proportions.aggregate(min)
  if criteria == 'std':
    selection = proportions.aggregate(np.std)
  selection = selection.sort_values(ascending = False)
  # Keep all jobs or subset by list
  keep_jobs = list(selection.index)
  keep_jobs.extend(['gender', 'category'])
  reg_df = subset_df[keep_jobs]
  # Replace spaces
  for token in reg_df.columns:
    if ' ' in token:                                                              
      old_token = token
      token = old_token.replace(" ", "_")
      reg_df = reg_df.rename(columns = {old_token : token})
    # Replace hyphens
    if '-' in token:                                                              
      old_token = token
      token = old_token.replace("-", "_")
      reg_df = reg_df.rename(columns = {old_token : token})
  return reg_df

def run_regression_main_effects(input_df, token, leave_out_cat, leave_out_gend):
  ''' Function to run regression main effects (no intersections) for ONE JOB and return model output'''
  df = input_df[['category', 'gender', token]]
  df = df.rename(columns = {'category': '**', 'gender': 'gender'})
  df = pd.get_dummies(df, columns = ['**', 'gender'])
  df.columns = df.columns.str.lstrip('**_')
  df = df.rename(columns = {'gender_W': 'woman', 'gender_M': 'man'})
  df = df.drop(columns = [leave_out_cat, leave_out_gend])
  X = df.loc[:, df.columns != token]
  y = df.loc[:, df.columns == token][token]
  y_columns = [y.name]
  X_columns = list(X.columns.values)
  print(f"""
  y_columns: {y_columns}
  X_columns: {X_columns}\n""")
  # METHOD: statsmodels.formula.api.logit
  main_effects = list(X.columns.values)
  all_regression_terms = f"{' + '.join(main_effects)}"
  print(all_regression_terms)
  logit_model = smf.logit(formula = f"{token} ~ {all_regression_terms}", data = df)
  try:
    est = logit_model.fit()
    print(f"\n> Logistic Regression {est.summary2()}")
    print("\n====================================================================\n\n\n")
    return est, main_effects, token
  except:
    print("\n\n******CANNOT FIT MODEL*******\n\n")
    pass

def run_regression_interaction(input_df, token, leave_out_cat, leave_out_gend):
  ''' Function to run regression main effects with intersections for ONE JOB and return model output'''
  df = input_df[['category', 'gender', token]]
  df = df.rename(columns = {'category': '**', 'gender': 'gender'})
  df = pd.get_dummies(df, columns = ['**', 'gender'])
  df.columns = df.columns.str.lstrip('**_')
  df = df.rename(columns = {'gender_W': 'woman', 'gender_M': 'man'})
  df = df.drop(columns = [leave_out_cat, leave_out_gend])
  X = df.loc[:, df.columns != token]
  y = df.loc[:, df.columns == token][token]
  y_columns = [y.name]
  X_columns = list(X.columns.values)
  print(f"""
  y_columns: {y_columns}
  X_columns: {X_columns}\n""")
  # METHOD: statsmodels.formula.api.logit
  main_effects = list(X.columns.values)
  interaction_terms = [list(X.columns.values)[-1] + ':' + cont for cont in list(X.columns.values)[:-1]]
  all_regression_terms = f"{' + '.join(main_effects)}" + " + " + f"{' + '.join(interaction_terms)}"
  logit_model = smf.logit(formula = f"{token} ~ {all_regression_terms}", data = df)
  try:
    est = logit_model.fit()
    print(f"\n> Logistic Regression {est.summary2()}")
    print("\n====================================================================\n\n\n")
    return est, main_effects, interaction_terms, token
  except:
    print("\n\n******CANNOT FIT MODEL*******\n\n")
    pass

def run_regression_no_woman(input_df, token, leave_out_cats = 'base', leave_out_genders = ['man', 'woman']):
  ''' Function to run regression main effects without woman dummy for ONE JOB and return model output'''
  df = input_df[['category', 'gender', token]]
  df = df.rename(columns = {'category': '**', 'gender': 'gender'})
  df = pd.get_dummies(df, columns = ['**', 'gender'])
  df.columns = df.columns.str.lstrip('**_')
  df = df.rename(columns = {'gender_W': 'woman', 'gender_M': 'man'})
  drops = [leave_out_cats]
  drops.extend(leave_out_genders)
  df = df.drop(columns = drops)
  X = df.loc[:, df.columns != token]
  y = df.loc[:, df.columns == token][token]
  y_columns = [y.name]
  X_columns = list(X.columns.values)
  print(f"""
  y_columns: {y_columns}
  X_columns: {X_columns}\n""")
  # METHOD: statsmodels.formula.api.logit
  main_effects = list(X.columns.values)
  all_regression_terms = f"{' + '.join(main_effects)}"
  print(all_regression_terms)
  logit_model = smf.logit(formula = f"{token} ~ {all_regression_terms}", data = df)
  try:
    est = logit_model.fit()
    print(f"\n> Logistic Regression {est.summary2()}")
    print("\n====================================================================\n\n\n")
    return est, main_effects, token
  except:
    print("\n\n******CANNOT FIT MODEL*******\n\n")
    pass

def initialize_regression(regression_df, regression_func = "main effects"):
  ''' Function to initialize regressions for ALL JOBS > THRESHOLD, with specified regression function
  Returns list of model results and list of professions'''
  leave_out_cat_list = ['base']
  leave_out_gend_list = ['man']
  models = []
  main_effects = []
  interactions = []
  models_professions = []
  i = 0
  jobs_list = list(regression_df.columns)
  jobs_list.remove('gender')
  jobs_list.remove('category')
  for TK in jobs_list:
    j = 0
    for LV_cat in leave_out_cat_list:
      for LV_gend in leave_out_gend_list:
        print(f"""
  _______________________________________   regression {i}.{j}
  _______________________________________   token: {TK}
  _______________________________________   leave_out_cat: {LV_cat}
  _______________________________________   leave_out_gend: {LV_gend}""")
      if regression_func == 'main effects':
        results = run_regression_main_effects(regression_df,token = TK, leave_out_cat = LV_cat, leave_out_gend = LV_gend)
        try:
          est, main_effects, token = results
          models.append(est)
          main_effects.append(main_effects)
          models_professions.append(token)
        except:
          print("\n\n******NO MODEL*******\n\n")
          pass
        j += 1
      if regression_func == 'no woman':
        results = run_regression_no_woman(regression_df,token = TK, leave_out_cats = 'base' ,leave_out_genders = ['man', 'woman'])
        try:
          est, main_effects, token = results
          models.append(est)
          main_effects.append(main_effects)
          models_professions.append(token)
        except:
          print("\n\n******NO MODEL*******\n\n")
          pass
        j += 1
      if regression_func == 'interactions':
        results = run_regression_interaction(regression_df,token = TK, leave_out_cat = LV_cat, leave_out_gend = LV_gend)
        try:
          est, main_effects, interaction_terms,  token = results
          models.append(est)
          main_effects.append(main_effects)
          interactions.append(interaction_terms)
          models_professions.append(token)
        except:
          print("\n\n******NO MODEL*******\n\n")
          pass
        j += 1
    i += 1
  return models, models_professions


def regression_results(models, models_professions, models_gen, models_professions_gen, models_int, models_professions_int):
  ''' Function to return key regression metrics: coeffs, R2 and P-Values.
  Summaries across three regression hierarchies: 
  (i) main effect - woman dummy
  (ii) all main effects
  (iii) all main effects + interactions'''
  # R2
  R2_main_effects= {}
  R2_gender = {}
  R2_interactions = {}
  # Coeff
  coeff_main_effects = {}
  coeff_interactions = {}
  coeff_gender = {}
  # P-Values
  p_interactions = {}
  # Main Effects
  for i in range(len(models)):
    profession = models_professions[i]
    R2 = (models[i].prsquared*100)
    coeffs = models[i].params
    R2_main_effects[profession] = R2
    coeff_main_effects[profession] = coeffs
  # No Woman
  for i in range(len(models_gen)):
    profession = models_professions_gen[i]
    R2 = (models_gen[i].prsquared*100)
    coeffs = models_gen[i].params
    R2_gender[profession] = R2
    coeff_gender[profession] = coeffs
  # Woman + Interactions
  for i in range(len(models_int)):
    profession = models_professions_int[i]
    R2 = (models_int[i].prsquared*100)
    coeffs = models_int[i].params
    p_values = models_int[i].pvalues
    R2_interactions[profession] = R2
    coeff_interactions[profession] = coeffs
    p_interactions[profession] = p_values
  # Output Results
  #R2
  R2_main_df = pd.DataFrame.from_dict(R2_main_effects, orient = 'index', columns = {'R2_main'})
  R2_int_df = pd.DataFrame.from_dict(R2_interactions, orient = 'index',  columns = {'R2_int'})
  R2_gen_df = pd.DataFrame.from_dict(R2_gender, orient = 'index',  columns = {'R2_gen'})
  # Coeffs
  coeff_main_df = pd.DataFrame.from_dict(coeff_main_effects)
  coeff_int_df = pd.DataFrame.from_dict(coeff_interactions)
  coeff_gen_df = pd.DataFrame.from_dict(coeff_interactions)
  # P-Values
  pvalue_df = pd.DataFrame.from_dict(p_interactions)
  return R2_main_df, R2_int_df, R2_gen_df, coeff_main_df, coeff_int_df, coeff_gen_df, pvalue_df


def count_signif_results(pvalue_df):
  ''' Function to return mean count and bool dataframe of signif p-values'''
  # Count P-Value < 0.05
  bool_pvalue_df = pvalue_df < 0.05
  mean = bool_pvalue_df.mean(axis = 1)
  # Count Negative, Positive
  return bool_pvalue_df, mean


def return_results_tables(freq_matrix_idents, freq_matrix_names, intersection, criteria, THRESHOLD):
  '''Function to run regressions and return regression results across three regression hierarchies: 
  (i) main effect - woman dummy
  (ii) all main effects
  (iii) all main effects + interactions
  '''

  regression_df = make_regression_df(freq_matrix_idents, freq_matrix_names, intersection, criteria, THRESHOLD)
  models, models_professions = initialize_regression(regression_df, 'main effects')
  models_int, models_professions_int = initialize_regression(regression_df, 'interactions')
  models_gen, models_professions_gen = initialize_regression(regression_df, 'no woman')
  R2_main_df, R2_int_df, R2_gen_df, coeff_main_df, coeff_int_df, coeff_gen_df, pvalue_df = regression_results(models, models_professions,
                                                                                                              models_gen, models_professions_gen,
                                                                                                              models_int, models_professions_int)                                             
  # R2 Table
  merge_1 = R2_gen_df.merge(R2_main_df, left_index = True, right_index = True)
  merge_2 = merge_1.merge(R2_int_df, left_index = True, right_index = True)
  merge_2['R2_gen_change'] = merge_2['R2_main'] - merge_2['R2_gen']
  merge_2['R2_int_change'] = merge_2['R2_int'] - merge_2['R2_main']
  R2 = merge_2.copy()
  # P Values
  bool_pvalue_df, mean_signif = count_signif_results(pvalue_df)
  # Count Signif
  value_df = coeff_int_df.copy()
  # If coeff = insignificant sets equal to 0, otherwise keeps coefficient
  for j in bool_pvalue_df.columns:
    for i in bool_pvalue_df.index:
      signif = bool_pvalue_df.loc[i, j]
      if signif == True:
        value_df.loc[i,j] = coeff_int_df.loc[i,j]
      else:
        value_df.loc[i,j] = 0
  return R2, bool_pvalue_df, value_df, mean_signif


## Heatmap of results
def make_heatmaps_coeffs(tabs_path, figs_path, intersections):
  ''' Function to create heatmap of significant coefficients'''
  fig, axes = plt.subplots(nrows=4,ncols=1, figsize = (35, 40))
  ax = axes.ravel()
  sns.set(font_scale=1.8)
  for i, intersection in enumerate(intersections):
    coeff_df = pd.read_csv(f'{tabs_path}/regression_results/{intersection}_coeff.csv', index_col = 0) 
    coeff_df.index = coeff_df.index.str.replace('female', 'woman')
    sns.heatmap(coeff_df, cmap = 'RdBu', center=0, ax = ax[i], vmin = -10, vmax = 10)
    ax[i].set_title(f'{intersection.upper()}', fontsize = 20, fontweight = 'bold')
    ax[i].tick_params(axis = 'x')
    ax[i].tick_params(axis = 'y', labelrotation = 0, labelsize = 20)
  plt.tight_layout()
  fig.savefig(f'{figs_path}/coeffs_heatmap.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

def make_heatmaps_pvalues(tabs_path, figs_path, intersections):
  ''' Function to create heatmap of significant p-values'''
  fig, axes = plt.subplots(nrows=4,ncols=1, figsize = (32,40))
  ax = axes.ravel()
  sns.set(font_scale=1.8)
  for i, intersection in enumerate(intersections):
    pvalue_df = pd.read_csv(f'{tabs_path}/regression_results/{intersection}_pvalue.csv', index_col = 0) 
    pvalue_df.index = pvalue_df.index.str.replace('female', 'woman')
    sns.heatmap(pvalue_df, cmap = 'binary', cbar = False, ax = ax[i])
    ax[i].set_title(f'{intersection.upper()}', fontsize = 20, fontweight = 'bold')
    ax[i].tick_params(axis = 'x')
    ax[i].tick_params(axis = 'y',labelrotation = 0, labelsize = 20)
  plt.tight_layout()
  fig.savefig(f'{figs_path}/pvalues_heatmap.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

def make_heatmaps_R2(tabs_path, figs_path, intersections):
  ''' Function to create heatmap of changes to R2'''
  fig, axes = plt.subplots(nrows=4, figsize = (35, 40))
  ax = axes.ravel()
  sns.set(font_scale=1.8)
  for i, intersection in enumerate(intersections):
    R2_df = pd.read_csv(f'{tabs_path}/regression_results/{intersection}_R2.csv', index_col = 0) 
    R2_changes = R2_df.drop(['R2_gen', 'R2_main', 'R2_int'], axis=1)
    R2_changes = R2_changes.rename(columns = {'R2_gen_change': 'Add Woman Dummy', 'R2_int_change': 'Add Interactions'})
    sns.heatmap(R2_changes.T, cmap = 'Reds', cbar = True, ax = ax[i], vmin = 0, vmax = 12)
    ax[i].set_title(f'{intersection.upper()}', fontsize = 20, fontweight = 'bold')
    ax[i].tick_params(axis = 'y', labelrotation = 0)
  plt.tight_layout()
  fig.savefig(f'{figs_path}/R2_heatmap.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

"""# Main Script"""

# SET PATH
def main():
  PATH = './BIAS_OUT_THE_BOX'
  model =  'GPT-2' #, 'XLNET']:
  data_path = f"{PATH}/data/{model}"
  figs_path = f"{PATH}/figs/{model}"
  tabs_path = f"{PATH}/tabs/{model}"

  freq_matrix_idents = pd.read_csv(f"{data_path}/{model}_freq_matrix_identity.csv", index_col = 0)
  freq_matrix_names = pd.read_csv(f"{data_path}/{model}_freq_matrix_names.csv", index_col = 0)

  threshold_dict = threshold_tables(freq_matrix_idents, freq_matrix_names)

  # Run regressions and summarize results
  R2_dfs = []
  pvalues_dfs = []
  means_dfs = []
  coeffs_dfs = []
  reg_results_dfs = []

  # Loop through intersections
  intersections = ['ethnicity', 'religion', 'sexuality', 'political']
  for intersection in intersections:
    # Output regression results
    R2, pvalue, coeff, mean_signif = return_results_tables(freq_matrix_idents, freq_matrix_names, intersection, 'range', threshold_dict[intersection])

    # R2
    R2.to_csv(f'{tabs_path}/regression_results/{intersection}_R2.csv')
    R2_dfs.append(R2)

    # P-values
    pvalue.to_csv(f'{tabs_path}/regression_results/{intersection}_pvalue.csv')
    pvalues_dfs.append(pvalue)

    # Mean signif
    pvalue.to_csv(f'{tabs_path}/regression_results/{intersection}_mean_signif.csv')
    means_dfs.append(mean_signif)

    # Coeffs
    coeff.to_csv(f'{tabs_path}/regression_results/{intersection}_coeff.csv')
    coeffs_dfs.append(coeff)

    # Generate summary table of coeffs + pvalues
    p = pvalue.copy()
    p.columns = [str(x) + ':p<0.05' for x in p.columns]
    c = coeff.copy()
    c.columns = [str(x) + ':signif_coeff' for x in c.columns]
    reg_results = pd.concat([p, c], axis = 1)
    reg_results = reg_results.reindex(sorted(reg_results.columns), axis=1)
    reg_results_dfs.append(reg_results)
    reg_results.to_csv(f'{tabs_path}/regression_results/{intersection}_logistic_regressions.csv')

  # Create summary results table
  frames = []
  for intersection, pvalue, mean, R2 in zip(intersections, pvalues_dfs, means_dfs, R2_dfs):
    cols = ['Intersection', 'Job Regs', 'Variable', 'Pct_Signif', 'Gen_R2_Change', 'Int_R2_Change']
    series = {'pct_signif': mean}
    results_df = pd.DataFrame(series)
    results_df['Intersection'] = intersection
    results_df['Job Regs'] = pvalue.shape[1]
    mean_gen_change = R2['R2_gen_change'].mean()
    mean_int_change = R2['R2_int_change'].mean()
    results_df['Gen_R2_Change'] = mean_gen_change
    results_df['Int_R2_Change'] = mean_int_change
    results_df = results_df.reset_index().rename(columns = {'index': 'Variable', 'pct_signif': 'Pct_Signif'})
    results_df = results_df[cols]
    frames.append(results_df)
  all_results = pd.concat(frames)
  all_results.to_csv(f'{tabs_path}/regression_results/all_regression_results.csv')

  # Just woman + interactions
  woman_results = all_results[all_results['Variable'].str.startswith('woman')]
  woman_results.to_csv(f'{tabs_path}/regression_results/woman_regression_results.csv')

  # Heat maps
  make_heatmaps_coeffs(tabs_path, figs_path, intersections)
  make_heatmaps_pvalues(tabs_path, figs_path, intersections)
  make_heatmaps_R2(tabs_path, figs_path, intersections)

if __name__ == "__main__":
  main()