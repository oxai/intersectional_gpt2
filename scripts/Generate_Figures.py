# -*- coding: utf-8 -*-
"""Generate_Figures.py

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
from adjustText import adjust_text
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pylab import *
import adjustText

"""# Functions

## Distributional Analysis
"""

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


## Distribution Plots
def make_rank_plot_all(figs_path, model, freq_matrix, intersections, logy = True, logx = True):
  '''Creates distribution plot for intersectional categories from ranks and cumulative share'''
  fig, ax = plt.subplots(1,1, figsize = (20, 6))

  stored_values = {}
  # Plot average distribution
  for gender in ['W', 'M']:
      color_dict = {'W': 'red', 'M': 'blue'}
      label_dict = {'W': 'Women (Average)', 'M': 'Men (Average)'}
      average_df, avg_bar_points = make_count_df(freq_matrix, gender, 'average', 0.5)
      pct_bar_points = make_count_df(freq_matrix, gender, 'average', 0.9)[1]
      if gender == 'W':
        linestyle = ':'
      else:
        linestyle = '-'

      average_df.plot('rank', 'share', logx = logx, logy = logy, ax = ax, label = label_dict[gender], color = color_dict[gender], alpha = 1, lw = 5, linestyle = linestyle)
      # Print labels and cumulative share of 50, 90 percentile
      print(label_dict[gender], avg_bar_points)
      stored_values[f'{gender}_avg'] = avg_bar_points[0]
      print(label_dict[gender], pct_bar_points)
      stored_values[f'{gender}_pct'] = pct_bar_points[0]

      # Plot 50 pct
      x_avg_line = (avg_bar_points[0], avg_bar_points[0])
      y_avg_line = (0, avg_bar_points[1])
      ax.plot(x_avg_line, y_avg_line, '--', alpha=1, color = color_dict[gender])

      # Plot 90 pct
      x_pct_line = (pct_bar_points[0], pct_bar_points[0])
      y_pct_line = (0, pct_bar_points[1])
      ax.plot(x_pct_line, y_pct_line, '--', alpha=1, color = color_dict[gender])
  
  # Plot intersectional lines
  maxes = []
  for intersection in intersections:
    for gender in ['W', 'M']:
      if gender == 'W':
        linestyle = ':'
      else:
        linestyle = '-'
      count_df, bar_points = make_count_df(freq_matrix, gender, intersection, 0.5)
      label = "_".join([intersection, gender])
      gender_labels = {'W': 'Woman', 'M': 'Man'}
      color_dict = {'W': 'red', 'M': 'blue'}
      count_df.plot('rank', 'share', logx = logx, logy = logy, ax = ax, color = color_dict[gender], alpha = 0.3, linestyle = linestyle)
      maxes.append(count_df['share'].max())
  handles, labels = ax.get_legend_handles_labels()
  ax.get_legend().remove()
  fig.legend(handles[0:2], labels[0:2], fontsize = 16, ncol = 2, loc = 'upper center', bbox_to_anchor=(0.54, 0.96))
  ax.set_xlabel("Log(Rank)", fontsize = 16)
  ax.set_ylabel("Share of Total", fontsize = 16)
  ax.set_ylim(0, (max(maxes)+0.01))
  ax.tick_params(axis = 'x', labelsize = 14)
  ax.tick_params(axis = 'y', labelsize = 14)

  # Annotations
  F_bar_points = make_count_df(freq_matrix, 'W', 'average', 0.5)[1]
  F_pct_points = make_count_df(freq_matrix, 'W', 'average', 0.9)[1]
  M_bar_points = make_count_df(freq_matrix, 'M', 'average', 0.5)[1]
  M_pct_points = make_count_df(freq_matrix, 'M', 'average', 0.9)[1]

  # Annotate 50%
  ax.annotate(f"{stored_values['M_avg']} jobs account for 50% of men",
          xy=(M_bar_points[0], M_bar_points[1]), xycoords='data',
          xytext=(0.43, 0.5), textcoords='axes fraction',
          arrowprops=dict(facecolor='black', width = 0.2, headwidth = 10),
          horizontalalignment='left', verticalalignment='top', fontsize = 14)
  
  ax.annotate(f"{stored_values['W_avg']} jobs account for 50% of women",
        xy=(F_bar_points[0], F_bar_points[1]), xycoords='data',
        xytext=(0.335, 0.6), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', width = 0.2, headwidth = 10),
        horizontalalignment='left', verticalalignment='top', fontsize = 14)
  
  # Annotate 90%
  ax.annotate(f"{stored_values['M_pct']} jobs account for 90% of men",
        xy=(M_pct_points[0], M_pct_points[1]), xycoords='data',
        xytext=(0.624, 0.1), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', width = 0.2, headwidth = 10),
        horizontalalignment='left', verticalalignment='top', fontsize = 14)
  
  ax.annotate(f"{stored_values['W_pct']} jobs account for 90% of women",
        xy=(F_pct_points[0], F_pct_points[1]), xycoords='data',
        xytext=(0.565, 0.2), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', width = 0.2, headwidth = 10),
        horizontalalignment='left', verticalalignment='top', fontsize = 14)

  plt.tight_layout()
  fig.savefig(f'{figs_path}/{model}_distributions_labelled.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

  return stored_values

def make_rank_plot_subplots(figs_path, model, freq_matrix_idents, freq_matrix_names, intersections, logy = True, logx = True):
  '''Creates distribution subplots for M,W for each intersection from ranks and cumulative share'''
  fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12,12))
  ax = axes.ravel()
  maxes = []
  for i, intersection in enumerate(intersections):
      for gender in ['W', 'M']:
        color_dict = {'W': 'red', 'M':'blue'}
        if intersection == 'continent':
          count_df = make_count_df(freq_matrix_names, gender, intersection, 0.5)[0]
        else:
          count_df= make_count_df(freq_matrix_idents, gender, intersection, 0.5)[0]
        if gender == 'W':
          linestyle = ':'
        else:
          linestyle = '-'
        label = "_".join([intersection, gender]) 
        count_df.plot('rank', 'share', logx = logx, logy = logy, ax = ax[i], label = label, color = color_dict[gender], linestyle = linestyle)
        maxes.append(count_df['share'].max())
        ax[i].set_xlabel("Log(Rank)", fontsize = 12)
        ax[i].set_ylabel("Share of Total", fontsize = 12)
        ax[i].set_ylim(0, (max(maxes)+0.01))
        # *****************
        print(max(maxes))
        print(maxes)
        ax[i].legend(fontsize = 12)
  plt.tight_layout()
  fig.savefig(f'{figs_path}/{model}_distributions_subplots.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

## Lorenz Curve and Gini
def plot_lorenz_curve(figs_path, model, freq_matrix_idents, freq_matrix_names, intersections, zoom = False):
  '''Creates lorenz curve plot for base M, base W from ranks and cumulative share'''
  fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12,12))
  ax = axes.ravel()
  for i, intersection in enumerate(intersections):
    for gender in ['W', 'M']:
      color_map = {'W':'red', 'M': 'blue'}
      if intersection == 'continent':
        count_df = make_count_df(freq_matrix_names, gender, intersection, 0.5)[0]
      else:
        count_df = make_count_df(freq_matrix_idents, gender, intersection, 0.5)[0]
      share_array = np.array(count_df['share'])
      X_lorenz = count_df['share'].cumsum() / count_df['share'].sum()
      label = "_".join([intersection, gender]) 
      if gender == 'W':
        markerstyle = 'x'
      else:
        markerstyle = 'o'
      ax[i].scatter(X_lorenz, np.arange(X_lorenz.size)/(X_lorenz.size-1), 
            marker=markerstyle, color=color_map[gender], s=20, label = label)
      # Set zoom to focus on smaller y-axis range
      if zoom == True:
        ax[i].set_ylim([-0.01, 0.2])
      else:
        ax[i].set_ylim([-0.05,1.05])
        ax[i].plot([0,1], [0,1], ls='--', color = 'black')
      ax[i].set_ylabel('Cumulative Share of Jobs', fontsize = 12)
      ax[i].set_xlabel('Cumulative Share of Total Workers', fontsize = 12)
      ax[i].legend(fontsize = 12)
  plt.tight_layout()
  fig.savefig(f'{figs_path}/{model}_lorenz_curve_subplots_zoom_{zoom}.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def return_gini(freq_matrix_idents, gender, intersection):
  '''Calculates gini_coefficient for gender-intersection pair'''
  count_df, bar_points = make_count_df(freq_matrix_idents, gender, intersection, 0.5)
  share_array = np.array(count_df['share'])
  return gini(share_array) 

def gini_table(freq_matrix_idents, intersections):
  '''Returns dataframe of gini coefficients'''
  gender_list = []
  intersection_list = []
  gini_list = []
  for gender in ['W', 'M']:
    for intersection in intersections:
      gini = return_gini(freq_matrix_idents, gender, intersection)
      gender_name = {'W':'Woman', 'M': 'Man'}
      gender_list.append(gender_name[gender])
      intersection_list.append(intersection.capitalize())
      gini_list.append(np.round(gini,3))
  results = pd.DataFrame({'gender':gender_list, 'intersection':intersection_list, 'gini_coeff':gini_list})
  results = results.sort_values(by = 'gini_coeff', ascending = True)
  return results

"""## Returned Occupations Analysis"""

def select_plot_df(freq_matrix, intersection, criteria, THRESHOLD, TOP_N):
  '''Subsets dataframe for subsequent plots

  Args:
    freq_matrix: input one-hot encoded job matrix
    intersection: category for analysis from ['base', 'ethnicity', 'religion', 'sexuality', 'political', 'continent']
    criteria: method of selecting large man-woman difference by min-max range ('range') or standard deviation ('std')
    THRESHOLD: lower-bound of mentions to exclude infrequently mentioned jobs'
    '''
  # Subset dataframe by intersection
  if intersection == 'continent':
    subset_df = freq_matrix
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
  # Apply threshold
  for token in df_frequency.columns:
    if df_frequency[token].aggregate(sum) < THRESHOLD:
      thresholds = thresholds.drop(columns = token)
  proportions = thresholds.div(thresholds.aggregate(sum))
  print(f'Number of categories: {proportions.shape[0]}, Number of Jobs > THRESHOLD: {proportions.shape[1]}')
  # Selection criteria
  if criteria == 'range':
    selection = proportions.aggregate(max) - proportions.aggregate(min)
  if criteria == 'std':
    selection = proportions.aggregate(np.std)
  # Apply TOP-N and sort
  if TOP_N == 'NONE':
    selection = selection.sort_values(ascending = False)
  else:
    selection = selection.sort_values(ascending = False)
    selection = selection.nlargest(TOP_N)

  # Make dataFrame
  plot_df = proportions[list(selection.index)]
  idx_ref = plot_df.groupby('gender').aggregate(sum).loc['M'].sort_values(ascending = False).index
  plot_df = plot_df.loc[:, (idx_ref)]

  return plot_df, proportions.shape[1]

def gender_parity_bar(figs_path, model, freq_matrix, criteria, THRESHOLD):
  '''Creates barplot of top man-woman range for num_jobs under THRESHOLD, selected by range or std criteria'''
  # Select plot data
  plot_df, num_jobs = select_plot_df(freq_matrix, 'base', criteria, THRESHOLD, TOP_N = 'NONE')
  plot_df = plot_df.aggregate(np.diff).aggregate(sum)
  # Set up subplots
  N_PLOTS = 1
  labels_plot = plot_df.index
  fig, ax = plt.subplots(nrows = N_PLOTS, figsize = (20, 3))
  x_TICK_NUMBER = list(range(num_jobs))
  print(f"Caption title:\nGender parity of {num_jobs} jobs.\nThreshold = {THRESHOLD}, criteria = {criteria}")
  # Plot bar
  ax.bar(x_TICK_NUMBER, plot_df.values, color = 'black', alpha = 0.7, zorder = 5)
  plt.sca(ax)
  plt.xticks(range(num_jobs), color = 'grey')
  plt.yticks([-1,0,1], labels = ['100% Men', 'Gender Parity', '100% Women'], fontsize = 14)          
  ax.grid(which = 'both', color = 'lightgrey', alpha = 0.5)
  ax.set_xlim((-1,num_jobs))
  ax.set_ylim((-1.05, 1.05))
  plt.hlines(0, -0.5, num_jobs-0.5, colors = 'red', linestyles = 'dashdot', zorder = 6)      # label = 'gender parity line',
  plt.text(0, 0.2, "Male-dominated jobs", color = 'red', fontsize = 14)
  plt.text(num_jobs-10, -0.2, "Female-dominated jobs", color = 'red', fontsize = 14)
  plt.xticks(x_TICK_NUMBER, labels_plot, rotation = 90, fontsize = 14, color = 'black')
  fig.savefig(f'{figs_path}/{model}_gender_parity.pdf', format='pdf', dpi=900, bbox_inches='tight')
  plt.show()

def stacked_bar(figs_path, model, freq_matrix, intersection, criteria, THRESHOLD):
  ''' Creates stacked-bar chart by intersection for num_jobs under THRESHOLD, selected by range or std criteria'''
  # Select plot data
  plot_df, num_jobs = select_plot_df(freq_matrix, intersection, criteria, THRESHOLD, TOP_N = 'NONE')
  # Set up subplots
  N_PLOTS = 2
  plot_labels = plot_df.columns.tolist()
  fig, axes = plt.subplots(nrows = N_PLOTS, figsize = (20, 8))
  x_TICK_NUMBER = list(range(num_jobs))
  print(f"Caption title:\nTop {num_jobs} jobs by gender and {intersection.upper()}.\nThreshold = {THRESHOLD}, criteria = {criteria}\n")
  # Initialise caches 
  array_dict = {}
  labels = []  
  k = 0
  # Plot bar
  for gender in sorted(list(set(plot_df.reset_index()['gender']))):
    array_meta = np.zeros(num_jobs)
    array_dict[gender] = np.empty((0,num_jobs))
    j = 0
    for category in sorted(list(set(plot_df.reset_index()['category']))):
      label = category.capitalize()
      array_dict[gender] = np.append(array_dict[gender], np.array([plot_df.loc[category, gender].values]), axis=0)
      labels.append(label)
      axes[k].bar(x_TICK_NUMBER, array_dict[gender][j], bottom = array_meta, label = label, width = 0.85, alpha = 0.7, lw = 1, edgecolor = 'black', zorder = 6)
      array_meta = array_meta + array_dict[gender][j]
      j += 1
    k += 1
  # Set axes and legend
  genders = ['Man', 'Woman']
  for idx, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    plt.xticks(range(num_jobs), color = 'white')
    plt.yticks(np.arange(0, 1.05, 0.2))
    ax.grid(which = 'both', color = 'lightgrey', alpha = 0.5)
    ax.set_xlim((-1, num_jobs))
    ax.set_ylim((0, 1.05))
    ax.set_title(genders[idx], fontsize = 16)
    ax.tick_params(axis = 'y', labelsize = 14)
  handles, labels = ax.get_legend_handles_labels()
  if intersection == 'religion':
    fig.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper center')
  elif intersection == 'continent':
    fig.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper center')
  elif intersection == 'sexuality':
    labels = ['Lesbian/Gay', 'Straight']
    fig.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper center')
  else:
    fig.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper center')
  plt.sca(axes[N_PLOTS - 1])
  plt.xticks(x_TICK_NUMBER, plot_labels, rotation = 90, fontsize = 14, color = 'black')
  fig.savefig(f'{figs_path}/{model}_full_barplot_{intersection}.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

def make_bar_scatter(figs_path, model, freq_matrix, intersection, criteria, THRESHOLD, TOP_N=10):
  ''' Creates 3-subplots, with LHS = scatter, RHS = 2 x stacked bars of TOP-N jobs under THRESHOLD by man-woman range'''
  # Set up 3-part figure grid
  fig3 = plt.figure(constrained_layout=False, figsize = (10,7))
  gs = fig3.add_gridspec(ncols=10, nrows=7)
  f3_ax1 = fig3.add_subplot(gs[:, 0:6])
  f3_ax1.set_title('')
  f3_ax2 = fig3.add_subplot(gs[0:3, 6:10])
  f3_ax2.set_title('bar1')
  f3_ax3 = fig3.add_subplot(gs[3:6, 6:10])
  f3_ax3.set_title('bar2')
  gs.update(wspace=1.5, hspace=1.5)

  ## BAR PLOTS
  # Make plotting df
  plot_df, num_jobs = select_plot_df(freq_matrix, intersection, criteria, THRESHOLD, TOP_N = 10)
  # Set up subplots
  TOP_N = 10
  plot_labels = plot_df.columns.tolist()
  x_TICK_NUMBER = list(range(TOP_N))
  # Initialise caches
  array_dict = {}
  labels = []  
  bar_axes = [f3_ax2, f3_ax3]
  # Make bar plot
  k = 0
  for gender in sorted(list(set(plot_df.reset_index()['gender']))):
    array_meta = np.zeros(TOP_N)
    array_dict[gender] = np.empty((0,TOP_N))
    j = 0
    for category in sorted(list(set(plot_df.reset_index()['category']))):
      label = category.capitalize()
      array_dict[gender] = np.append(array_dict[gender], np.array([plot_df.loc[category, gender].values]), axis=0)
      labels.append(label)
      bar_axes[k].bar(x_TICK_NUMBER, array_dict[gender][j], bottom = array_meta, label = label, width = 0.85, alpha = 0.7, lw = 1, edgecolor = 'black', zorder = 6)
      array_meta = array_meta + array_dict[gender][j]
      j += 1
    k += 1

  # Axes
  genders = ['Man', 'Woman']
  for idx, ax in enumerate(bar_axes):
    plt.sca(ax)
    plt.xticks(range(TOP_N), color = 'white')
    plt.yticks(np.arange(0, 1.05, 0.2))
    ax.grid(which = 'both', color = 'lightgrey', alpha = 0.5)
    ax.set_xlim((-1, TOP_N))
    ax.set_ylim((0, 1.05))
    ax.set_title(genders[idx], fontsize = 16)
    ax.tick_params(axis = 'y', labelsize = 14)
  N_PLOTS = 2
  plt.sca(bar_axes[N_PLOTS - 1])
  plt.xticks(x_TICK_NUMBER, plot_labels, rotation = 90, fontsize = 16, color = 'black')

  ## SCATTER PLOT
  # Select plot data
  subset_df, num_jobs = select_plot_df(freq_matrix, intersection, criteria, THRESHOLD, TOP_N='NONE')
  subset_df = subset_df.reset_index()


  # Create list of jobs
  jobs_list = list(subset_df.columns)
  jobs_list.remove('gender')
  jobs_list.remove('category')
  frames = []
  for job in jobs_list:
    job_df = subset_df.pivot(index='category', columns='gender')[(job)].reset_index()
    job_df['job'] = job
    job_df = job_df.set_index('job')
    frames.append(job_df)
  all_jobs = pd.concat(frames)
  # Make scatter plot
  xs = all_jobs['W']
  ys = all_jobs['M']
  f3_ax1.set_xlabel("Over-representation Factor (Women)", fontsize = 16)
  f3_ax1.set_ylabel("Over-representation Factor (Men)", fontsize = 16)
  f3_ax1.set_xlim(-0.05,1)
  f3_ax1.set_ylim(-0.05,1)
  f3_ax1.tick_params(axis = 'x', labelsize = 14)
  f3_ax1.tick_params(axis = 'y', labelsize = 14)
  sns.scatterplot(x=xs, y=ys, data=all_jobs, hue='category', marker="x", ax = f3_ax1)

  # Annotations
  all_jobs['diff'] = all_jobs['M'] - all_jobs['W']
  all_jobs = all_jobs.sort_values(by ="diff", ascending = False)
  jobs_list = list(all_jobs.index)
  together = []
  # Woman-dominated jobs
  for i in range(-1, -5, -1):
    label = jobs_list[i]
    x = all_jobs['W'].iloc[i]
    y = all_jobs['M'].iloc[i]
    together.append((label,x,y))
  #Man-dominated jobs        
  for i in range(0, 4, 1):
    label = jobs_list[i]
    x = all_jobs['W'].iloc[i]
    y = all_jobs['M'].iloc[i]
    together.append((label, x,y))
  together.sort()
  text = [x for (x,y,z) in together]
  xs = [y for (x,y,z) in together]
  ys = [z for (x,y,z) in together]
  texts = []
  for x, y, s in zip(xs, ys, text):
      texts.append(f3_ax1.text(x, y, s, fontsize = 14))

  # Legend
  handles, labels = f3_ax1.get_legend_handles_labels()
  f3_ax1.get_legend().remove()
  labels = [s.capitalize() for s in labels]
  if intersection == 'religion':
    fig3.legend(handles, labels, fontsize = 16, ncol = 1, loc = 'upper right', bbox_to_anchor=(0.51, 0.84), borderaxespad=1)
  elif intersection == 'continent':
    fig3.legend(handles, labels, fontsize = 16, ncol = 1, loc = 'upper right',bbox_to_anchor=(0.51, 0.84), borderaxespad=1)
  elif intersection == 'sexuality':
    labels = ['Lesbian/Gay', 'Straight']
    fig3.legend(handles, labels, fontsize = 16, ncol = 1, loc = 'upper right', bbox_to_anchor=(0.51, 0.84), borderaxespad=1)
  elif intersection == 'political':
    fig3.legend(handles, labels, fontsize = 16, ncol = 1, loc = 'upper right',bbox_to_anchor=(0.51, 0.84), borderaxespad=1)
  else:
    fig3.legend(handles, labels, fontsize = 16, ncol = 1, loc = 'upper right', bbox_to_anchor=(0.51, 0.84), borderaxespad=1)

  # INLINE LABELS
  # if intersection == 'religion':
  #   fig3.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper left', bbox_to_anchor=(0, 1.1))
  # elif intersection == 'sexuality':
  #   labels = ['Lesbian/Gay', 'Straight']
  #   fig3.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper left', bbox_to_anchor=(0.2, 1.1))
  # elif intersection == 'political':
  #   fig3.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper center')
  # elif intersection == 'continent':
  #   fig3.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper left', bbox_to_anchor=(0, 1.1))
  # else:
  #   fig3.legend(handles, labels, fontsize = 16, ncol = len(labels), loc = 'upper left', bbox_to_anchor=(0.1, 1.02))
  # Lines
  x_line = [1*(1/len(labels)), 0]
  y_line = [0, 1*(1/len(labels))]
  f3_ax1.plot(x_line, y_line, '--', alpha=0.9, zorder=0, color = 'black')

  # Scale axis
  if intersection == 'religion':
    scale_factor = 5*(1/len(labels))
    f3_ax1.set_xlim(-0.05,scale_factor)
    f3_ax1.set_ylim(-0.05,scale_factor)
    f3_ax1.set_xticks([0, scale_factor/5, ((scale_factor/5) * 2), ((scale_factor/5) * 3), ((scale_factor/5) * 4), scale_factor])
    f3_ax1.set_xticklabels(['0x','1x', '2x', '3x', '4x', '5x'])
    f3_ax1.set_yticks([0, scale_factor/5, ((scale_factor/5) * 2), ((scale_factor/5) * 3), ((scale_factor/5) * 4), scale_factor])
    f3_ax1.set_yticklabels(['0x','1x', '2x', '3x', '4x', '5x'])
  elif intersection == 'ethnicity':
    scale_factor = 3*(1/len(labels))
    f3_ax1.set_xlim(-0.05,scale_factor)
    f3_ax1.set_ylim(-0.05,scale_factor)
    f3_ax1.set_xticks([0, scale_factor/3, ((scale_factor/3) * 2), scale_factor])
    f3_ax1.set_xticklabels(['0x','1x', '2x', '3x'])
    f3_ax1.set_yticks([0, scale_factor/3, ((scale_factor/3) * 2), scale_factor])
    f3_ax1.set_yticklabels(['0x','1x', '2x', '3x'])
  elif intersection == 'continent':
    scale_factor = 2*(1/len(labels))
    f3_ax1.set_xlim(-0.025,scale_factor)
    f3_ax1.set_ylim(-0.025,scale_factor)
    f3_ax1.set_xticks([0, scale_factor/2, scale_factor])
    f3_ax1.set_xticklabels(['0x','1x','2x'])
    f3_ax1.set_yticks([0, scale_factor/2, scale_factor])
    f3_ax1.set_yticklabels(['0x','1x','2x'])
  else:
    scale_factor = 2*(1/len(labels))
    f3_ax1.set_xlim(-0.05,scale_factor)
    f3_ax1.set_ylim(-0.05,scale_factor)
    f3_ax1.set_xticks([0, scale_factor/2, scale_factor])
    f3_ax1.set_xticklabels(['0x','1x','2x'])
    f3_ax1.set_yticks([0, scale_factor/2, scale_factor])
    f3_ax1.set_yticklabels(['0x','1x','2x'])


  adjust_text(texts, x=xs, y=ys, ax=f3_ax1, autoalign='y',only_move={'points':'xy', 'text':'xy'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
  fig3.savefig(f'{figs_path}/{model}_scatterbar_{intersection}.pdf', format='pdf', dpi=600, bbox_inches='tight')
  plt.show()

"""# Main Script"""

data_path = './BIAS_OUT_THE_BOX/data/GPT-2'
freq_matrix_idents = pd.read_csv(f"{data_path}/GPT-2_freq_matrix_identity.csv", index_col = 0)
freq_matrix_names = pd.read_csv(f"{data_path}/GPT-2_freq_matrix_names.csv", index_col = 0)

figs_path = './BIAS_OUT_THE_BOX/figs/XLNET'

def main():
  # SET PATH
  PATH = './BIAS_OUT_THE_BOX'

  for model in ['GPT-2','XLNET']:
    print(f'Generating results for {model}')
    print('####################\n\n')

    data_path = f"{PATH}/data/{model}"
    figs_path = f"{PATH}/figs/{model}"
    tabs_path = f"{PATH}/tabs/{model}"

    print(data_path)
    print(figs_path)

    freq_matrix_idents = pd.read_csv(f"{data_path}/{model}_freq_matrix_identity.csv", index_col = 0)
    freq_matrix_names = pd.read_csv(f"{data_path}/{model}_freq_matrix_names.csv", index_col = 0)

    threshold_dict = threshold_tables(freq_matrix_idents, freq_matrix_names)

    # Plot Distributions (one graph)
    intersections = ['base', 'sexuality', 'political','religion', 'ethnicity'] 
    stored_values = make_rank_plot_all(figs_path, model, freq_matrix_idents, intersections, logy=False, logx = True)

    # Plot Distributions (subgraph)
    intersections = ['base', 'sexuality', 'political','religion', 'ethnicity', 'continent'] 
    make_rank_plot_subplots(figs_path, model, freq_matrix_idents, freq_matrix_names, intersections, logy=False, logx = True)

    # Plot Lorenz Curve, zoomed out and zoomed in (subplots)
    intersections = ['base', 'sexuality', 'political','religion', 'ethnicity', 'continent'] 
    plot_lorenz_curve(figs_path, model, freq_matrix_idents, freq_matrix_names, intersections, zoom = False)  
    plot_lorenz_curve(figs_path, model, freq_matrix_idents, freq_matrix_names, intersections, zoom = True)

    # Calculate gini coefficients
    intersections = ['base', 'sexuality', 'political','religion', 'ethnicity']
    results = gini_table(freq_matrix_idents, intersections)
    results.index = pd.RangeIndex(start = 0, stop = len(results),step=1)
    # Reindex coefficients relative to base_M, base_W
    base_man = results[(results['gender']=='Man') & (results['intersection']=='Base')]
    man_index_value = 100/base_man['gini_coeff'].values
    results['relative_to_baseM'] = np.round(results['gini_coeff'] * man_index_value,3)
    # Export gini table
    results.to_csv(f'{tabs_path}/{model}_gini_results.csv')

    # Gender parity plot
    gender_parity_bar(figs_path, model, freq_matrix_idents, "range", threshold_dict['base'])

    # Stacked Bar plots per intersection
    intersections = ['sexuality', 'political','religion', 'ethnicity', 'continent']
    for intersection in intersections:
      if intersection == 'continent':
        stacked_bar(figs_path, model, freq_matrix_names, intersection, 'range', threshold_dict[intersection])
      else:
        stacked_bar(figs_path, model, freq_matrix_idents, intersection, 'range', threshold_dict[intersection])

    # Scatter Bar plots per intersection
    for intersection in intersections:
      if intersection == 'continent':
        make_bar_scatter(figs_path, model, freq_matrix_names, intersection, 'range', threshold_dict[intersection])
      else:
        make_bar_scatter(figs_path, model, freq_matrix_idents, intersection, 'range', threshold_dict[intersection])

# Run main
if __name__ == "__main__":
  main()