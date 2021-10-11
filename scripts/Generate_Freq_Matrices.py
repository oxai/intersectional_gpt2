# -*- coding: utf-8 -*-
"""Generate_Freq_Matrices.py
"""

"""# Imports and Installs"""

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from scipy import stats
from ast import literal_eval
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

""" # Functions"""

## Cleaning Functions
def clean_category_labels(i):
  '''Function for cleaning categorical labels'''
  if i == 'man':
    return 'Base man'
  elif i =='woman':
    return 'Base woman'
  elif i =='Native American woman':
    return 'Native-American woman'
  elif i == 'Native American man':
    return 'Native-American man'
  elif i == 'lesbian woman':
    return 'gay woman'
  else:
    return i

def clean_genders(i):
  '''Function for converting gender labels'''
  if i == 'man':
    return 'M'
  if i == 'woman':
    return 'W'


def clean_up_data(input_df, template = 'identity'):
  '''Function to clean sentence completition data'''
  df = input_df.copy()
  df.index = pd.RangeIndex(start = 0, stop = len(df), step = 1)

  if template == 'identity':
      # Remove categories not in NEURIPS paper
    removes = ['Catholic man', 'Catholic woman', 'Sikh woman', 'Sikh man', 'Indian woman', 'Indian man', 'Native American woman', 'Native American man']
    df = df[~df['Name'].isin(removes)]

    df['Gender'] = ''
    df['Category'] = ''
    df = df[['Name', 'Gender', 'Category', 'Title']]
    df['Name'] = df['Name'].map(lambda x: clean_category_labels(x))
    df[['Category', 'Gender']] = df.Name.str.split(expand=True,)
    df['Category'] = df['Category'].str.lower()
    df['Name'] = df['Name'].str.lower()
    df['Gender'] = df['Gender'].map(lambda x: clean_genders(x))

  elif template == 'names':

    df['Gender'] = ''
    df['Category'] = ''
    df = df[['Name', 'Gender', 'Category', 'Title']]
    #assert len(df) == 200000
    for index in range(len(df)):
        if index < 20000:
          df.loc[index, 'Category'] = 'Africa'
          df.loc[index, 'Gender'] = 'W'
        elif index < 40000:
          df.loc[index, 'Category'] = 'Americas'
          df.loc[index, 'Gender'] = 'W'
        elif index < 60000:
          df.loc[index, 'Category'] = 'Asia'
          df.loc[index, 'Gender'] = 'W'
        elif index < 80000:
          df.loc[index, 'Category'] = 'Europe'
          df.loc[index, 'Gender'] = 'W'
        elif index < 100000:
          df.loc[index, 'Category'] = 'Oceania'
          df.loc[index, 'Gender'] = 'W'
        elif index < 120000:
          df.loc[index, 'Category'] = 'Africa'
          df.loc[index, 'Gender'] = 'M'
        elif index < 140000:
          df.loc[index, 'Category'] = 'Americas'
          df.loc[index, 'Gender'] = 'M'
        elif index < 160000:
          df.loc[index, 'Category'] = 'Asia'
          df.loc[index, 'Gender'] = 'M'
        elif index < 180000:
          df.loc[index, 'Category'] = 'Europe'
          df.loc[index, 'Gender'] = 'M'
        else:
          df.loc[index, 'Category'] = 'Oceania'
          df.loc[index, 'Gender'] = 'M'
    df = df[df['Name']!= 'Princess']

  return df

## Missing data calculations
def calc_missing_title_count(input_df, col):
  ''' Function to calculate the proportion of sentence completions without a returned title'''
  df = input_df.copy()
  cat_labels = []
  missing_counts = []
  extracted_counts = []
  for cat in df[col].unique():
    cat_labels.append(cat)
    subset_df = df[df[col]==cat]
    total = len(subset_df)
    missing = len(subset_df[subset_df['Title'] == '[]'])
    missing_counts.append(missing)
    extracted = total - missing
    extracted_counts.append(extracted)

  return cat_labels, missing_counts, extracted_counts

def make_missing_plot(templates, cleaned_dfs, figs_path, model):
  width = 0.35       # the width of the bars: can also be len(x) sequence
  fig, ax = plt.subplots(2, 2, figsize = (12,6))
  cols = ['Category', 'Gender']
  base_axes = [0,1]
  for template, df, base_ax in zip(templates, cleaned_dfs, base_axes):
    for i, col in enumerate(cols):
      labels, missing_counts, extracted_counts = calc_missing_title_count(df, col)
      ax[base_ax][i].bar(labels, missing_counts, width, label='Missing')
      ax[base_ax][i].bar(labels, extracted_counts, width, bottom=missing_counts,
            label='Extracted')
      ax[base_ax][i].set_ylabel('Count', fontsize = 16)
      ax[base_ax][i].tick_params(axis = 'x', labelsize = 14)
      ax[base_ax][i].tick_params(axis = 'y', labelsize = 14)
      if col == 'Category':
        ax[base_ax][i].set_title(f'{template}-template; intersection', fontsize = 18)
        ax[base_ax][i].set_xticklabels([x.capitalize() for x in labels], rotation = 90)
        if template == 'identity':
          ax[base_ax][i].set_ylim(0,15000)
        else:
          ax[base_ax][i].set_ylim(0,40000)
      else:
        ax[base_ax][i].set_xticklabels(['Man', 'Woman'], rotation = 90)
        ax[base_ax][i].set_title(f'{template}-template; gender', fontsize = 18)
      
        
  handles, leg_labels = ax[0][0].get_legend_handles_labels()
  fig.legend(handles, leg_labels, bbox_to_anchor = (0.5, 1.025), loc='center', ncol = 2, fontsize = 16)
  plt.tight_layout()
  fig.savefig(f'{figs_path}/{model}_missing_titles.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()


## Frequency matrix generation
def load_job_replacements(PATH):# Load job replacement data
  job_replacements = pd.read_csv(f"{PATH}/data/shared_data/job_replacements.csv")
  job_replacements = job_replacements.dropna()
  job_replacements.index = pd.RangeIndex(start = 0, stop = len(job_replacements), step = 1)

  # Create column renaming dictionary
  replacements_dict = {}
  for i in range(len(job_replacements)):
    job = job_replacements['job'].iloc[i]
    update_match = job_replacements['update_match'].iloc[i]
    replacements_dict[job] = update_match

  return replacements_dict

def make_freq_matrix(input_df, replacements_dict):
  '''Function for converting raw tokens data to hot-encoded matrix for categories data'''
  df = input_df.copy()
  # Convert to list type
  df['Title'] = df['Title'].apply(literal_eval)
  df = df.explode('Title')
  # Create dummies
  dummies = pd.get_dummies(df['Title'])
  hot_df = df.merge(dummies, left_index = True, right_index = True).drop('Title', axis = 1)

  # Convert columns to lower case
  hot_df.columns = hot_df.columns.str.lower()

  # Rename columns
  hot_df = hot_df.rename(columns = (replacements_dict))

  # Aggregate duplicate columns
  hot_df = hot_df.groupby(axis=1, level=0).sum()

  # Reorder columns
  cols = list(hot_df)
  for col_name in ['gender', 'category', 'name']:
      cols.insert(0, cols.pop(cols.index(col_name)))
  hot_df = hot_df.loc[:, cols]

  return hot_df

PATH = './BIAS_OUT_THE_BOX'
model = 'XLNET'
data_path = f"{PATH}/data/{model}"
df = pd.read_csv(f"{data_path}/NER_output/identity_occupations_template.csv", index_col = 0)

df['Name'].value_counts()

"""# Script"""

def main():

  # SET PATH
  PATH = './BIAS_OUT_THE_BOX'

  for model in ['GPT-2', 'XLNET']:
    print(f'Loading sentence completions for {model}')
    print('####################\n\n')
    data_path = f"{PATH}/data/{model}"
    figs_path = f"{PATH}/figs/{model}"

    df_names = pd.read_csv(f"{data_path}/NER_output/names_occupations_template.csv", index_col = 0)
    df_idents = pd.read_csv(f"{data_path}/NER_output/identity_occupations_template.csv", index_col = 0)

    cleaned_names = clean_up_data(df_names, template = 'names')
    cleaned_idents = clean_up_data(df_idents, template = 'identity')

    templates = ['identity', 'names']
    cleaned_dfs = [cleaned_idents, cleaned_names]

    # Calculate missing titles
    make_missing_plot(templates, cleaned_dfs, figs_path, model)



    for template, cleaned_df in zip(templates, cleaned_dfs):

      print(f'Generating freq matrix for {template}-template')
      print('####################\n\n')

      # Load job replacements
      replacements_dict = load_job_replacements(PATH)

      freq_matrix = make_freq_matrix(cleaned_df, replacements_dict)

      # Save as csv
      freq_matrix.to_csv(f"{data_path}/{model}_freq_matrix_{template}.csv")


# Run main
if __name__ == "__main__":
  main()
