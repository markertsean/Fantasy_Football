import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import pickle as pkl
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
import sys

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = 14,10

#TODO: Proper implementation
sys.path.append('/home/sean/Documents/Fantasy_Football/')
from util import utilities
from model import model_generation
from normalize import normalize_raw_input

__analyze_version__ = '0.1.0'

continuous_offensive_points = {
    'rushing_yards':1/10.,
    'receiving_yards':1/10.,
    'passing_yards':1/25.,
    'complete_pass':0.5,
}
categorical_offensive_points = {
    'pass_touchdown':4,
    'receive_touchdown':6,
    'rush_touchdown':6,
    'td_yards_40':2,
    'fumble':-2,
    'offensive_interception':-1,
}
kicker_points = {
    'close_field_goal_success':3,
    'close_field_goal_failure':-2,
    'far_field_goal_success':4,
    'far_field_goal_failure':-1,
    'extra_point':1,
}
defensive_points = {
    'sack':1,
    'defensive_fumble_forced':3,
    'defensive_interception':2,
    'defensive_points_allowed':0,
}
def defensive_points_allowed(x):
    assert isinstance(x,int) # Input must be float
    if (x==0):
        return 10
    elif (x<7):
        return 7
    elif (x<14):
        return 4
    elif (x<21):
        return 1
    elif (x<28):
        return 0
    elif (x<35):
        return -1
    else:
        return -4
all_points_dict = {
    **continuous_offensive_points,
    **categorical_offensive_points,
    **kicker_points,
    **defensive_points,
}

def nCk(n, k):
    k=min(k,n-k)
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def nPk(n, k):
    k=min(k,n-k)
    return math.factorial(n) // math.factorial(n - k)

class range_class:
    def __init__(self):
        self.my_range = {}

    def get_first_digit(self,inp_str):
        start_ind=0
        for i in range(start_ind,len(inp_str)):
            start_ind = i
            if (inp_str[i].isdigit()):
                break
        end_ind = start_ind+1
        for i in range(end_ind,len(inp_str)):
            end_ind = i
            if (not inp_str[i].isdigit()):
                break
        return inp_str[start_ind:end_ind]

    def add_range(self,inp_str):
        first_number = self.get_first_digit(inp_str)
        if (inp_str[0] == '_'):
            self.my_range[int(first_number)] = inp_str[1:]
        elif (inp_str[-1] == '_'):
            self.my_range[int(first_number)] = inp_str[:-1]+"+"
        else:
            self.my_range[int(first_number)] = inp_str.replace('_','-')

    def get_range(self,inp_digit):
        this_digit = int(inp_digit)
        for digit in (sorted(self.my_range.keys()))[::-1]:
            if (this_digit>= digit):
                return self.my_range[digit]
        return None

    def get_range_dict(self):
        return self.my_range

def get_ranges(columns):
    cols_to_gen = []
    out_dict = {}
    for cat in [
        categorical_offensive_points,
        defensive_points,
    ]:
        for key in cat.keys():
            cols_to_gen.append(key)
            out_dict[cols_to_gen[-1]] = range_class()

    for col in columns:
        for good_col in cols_to_gen:
            if ((col.startswith(good_col)) and ("_range_" in col)):
                start_ind = col.index('_range_')+7
                out_dict[good_col].add_range(col[start_ind:])

    return out_dict

def get_analyze_path(version=__analyze_version__):
    return utilities.get_project_dir()+'data/analysis/'+version+'/'

def load_predictions( input_arguments ):

    full_file_path = model_generation.get_prediction_path(input_arguments["model_version"])+input_arguments['prediction_file_name']

    prediction_dict = {}
    with open(full_file_path,'rb') as f:
        inp_dict = pkl.load(f)

    return inp_dict


# Makes ranges from categories readable
def map_bucket_value(col,inp_df):
    run_map = False

    for val in inp_df[col].unique():
        try:
            int(val)
        except:
            run_map=True
            break
    if (not run_map):
        return inp_df[col]
    map_dict = {}
    for val in inp_df[col].unique():
        all_numbers_dash = [int(s) for s in val.split('-') if s.isdigit()]
        all_numbers_plus = [int(s) for s in val.split('+') if s.isdigit()]
        all_numbers_le = [int(s) for s in val.split('<=') if s.isdigit()]
        all_numbers = all_numbers_dash + all_numbers_plus + all_numbers_le
        sum_val = 0
        for i in range(0,len(all_numbers)):
            sum_val += all_numbers[i]*1.0
        map_dict[val] = sum_val/len(all_numbers)
    return inp_df[col].replace(map_dict)

def plot_pred_true(cols,inp_pred_df,inp_true_df,input_args):
    plot_output_path = get_analyze_path(input_args["analyze_version"])
    plot_full_fn_pre = plot_output_path+input_args['prediction_file_name'] + "/scatter/"
    os.makedirs(plot_full_fn_pre,exist_ok=True)

    pred_df = inp_pred_df.copy()
    true_df = inp_true_df.copy()
    if (isinstance(cols,str)):
        cols = [cols]
    for col in cols:
        plot_full_fn = plot_full_fn_pre + input_args['prediction_file_name']+"_" + col + ".png"

        pred_df[col] = map_bucket_value(col,pred_df)

        min_value = min(pred_df[col].astype(int).min(),true_df[col].astype(int).min())
        max_value = max(pred_df[col].astype(int).max(),true_df[col].astype(int).max())

        min_value = 0.9*min_value if min_value > 0 else (
            -1 if min_value == 0 else 1.1*min_value
        )
        max_value = 1.1*max_value if max_value > 0 else (
            1 if max_value == 0 else 0.9*max_value
        )

        plt.clf()
        plt.scatter(pred_df[col],true_df[col],color='r',alpha=0.05)
        plt.plot([min_value,max_value],[min_value,max_value],linestyle='--',color='b')
        plt.xlim([int(min_value),int(max_value)])
        plt.ylim([int(min_value),int(max_value)])
        plt.title(col+
            ":     R^2={:10.2e}     Explained Variance={:10.2e}".format(
                r2_score(true_df[col],pred_df[col]),
                explained_variance_score(true_df[col],pred_df[col])
            )
        )
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.savefig(plot_full_fn)
        print("Wrote "+plot_full_fn)

from sklearn.metrics import plot_confusion_matrix

def plot_confusion(cols,inp_pred_df,inp_true_df,input_args):
    plot_output_path = get_analyze_path(input_args["analyze_version"])
    plot_full_fn_pre = plot_output_path+input_args['prediction_file_name'] + "/confusion/"
    os.makedirs(plot_full_fn_pre,exist_ok=True)

    pred_df = inp_pred_df.copy()
    true_df = inp_true_df.copy()
    key_fields = ['season','week','team','opponent']
    if (isinstance(cols,str)):
        cols = [cols]
    for col in cols:
        plot_full_fn = plot_full_fn_pre + input_args['prediction_file_name']+"_" + col + ".png"

        if (pred_df[col].unique().shape[0] > 20):
            continue

        true_count = {}
        true_bins = sorted(true_df[col].unique())
        for t in true_bins:
            true_count[t] = true_df.loc[true_df[col]==t,col].shape[0]

        pred_count = {}
        pred_bins = sorted(pred_df[col].unique())
        for p in pred_bins:
            pred_count[p] = pred_df.loc[pred_df[col]==p,col].shape[0]

        all_bins = true_bins

        comb_df = true_df[key_fields+[col]].rename(
            columns={col:col+"_true"}
        ).merge( pred_df[key_fields+[col]].rename(
            columns={col:col+"_pred"}
        ),
            on=key_fields
        )

        conf = {}
        for t in all_bins:
            conf[t] = {}
            for p in all_bins:
                conf[t][p] = comb_df.loc[
                    (comb_df[col+"_true"]==t) &
                    (comb_df[col+"_pred"]==p)
                ].shape[0]

        inp_array = np.zeros([len(all_bins),len(all_bins)])
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for i_t in range(0,len(all_bins)):
            for i_p in range(0,len(all_bins)):
                inp_array[i_t,i_p] = conf[all_bins[i_t]][all_bins[i_p]]

                if (i_t == i_p):
                    tp += inp_array[i_t,i_p]
                else:
                    tn += inp_array[i_t,i_p]

        for i in range(0,len(all_bins)):
            fn += inp_array[i,:].sum() - inp_array[i,i]
            fp += inp_array[:,i].sum() - inp_array[i,i]

        micro_precision = tp / (tp+fp+1e-7)
        micro_recall = tp / (tp+fn+1e-7)

        all_bins = [ str(s) for s in all_bins ]

        df_cm = pd.DataFrame(
            inp_array,
            index = all_bins,
            columns = all_bins
        )

        plt.clf()
        sns.heatmap(df_cm, annot=True,fmt='g')
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.title("{:s}:    Micro Precision={:8.4f}".format(
            col,
            micro_precision,
        ))
        plt.savefig(plot_full_fn)
        print("Wrote "+plot_full_fn)

def breakdown_categories(
    inp_df,
    fields_to_parse,
    key_fields=['season','week','team','opponent'],
    drop_cols=True
):
    if (isinstance(fields_to_parse,str)):
        fields_to_parse = [fields_to_parse]

    out_df = inp_df.copy()

    for field_pattern in fields_to_parse:
        valid_fields = []
        for key in out_df.columns:
            if ( ( field_pattern in key ) and ('_range_' in key)):
                valid_fields.append(key)

        if (len(valid_fields)>0):
            num_value = {}
            str_value = {}
            for field in valid_fields:
                num_start = field.index('_range_')+7
                for i in range(num_start,len(field)):
                    num_start = i
                    if (field[i].isdigit()):
                        break
                num_end = num_start+1
                for i in range(num_end,len(field)):
                    num_end = i
                    if (not field[i].isdigit()):
                        break

                num_str = field[num_start:num_end]
                # Just a start num
                if (field[-1]=='_'):
                    str_value[field] = num_str+"+"
                # Just end num
                elif (field.index('_range_')+7=='_'):
                    str_value[field] = "<="+num_str
                else:
                    str_value[field] = field[num_start:].replace("_","-")

                num_value[field] = int(num_str)

            out_df[field_pattern] = ''

            out_df['best_field'] = out_df[valid_fields].idxmax(axis=1)
            out_df[field_pattern] = out_df['best_field'].apply(lambda x: str_value[x])
            out_df[field_pattern+'_predicted'] = out_df['best_field'].apply(lambda x: num_value[x])
            out_df = out_df.drop(columns=['best_field'])
            if drop_cols:
                out_df = out_df.drop(columns=valid_fields)

    return out_df

def get_val_from_range(inp_str):
    if (
            isinstance(inp_str,float) or
            isinstance(inp_str,int) or
            isinstance(inp_str,np.int64)
    ):
        return inp_str
    if (inp_str[-1]=='+'):
        return float(inp_str[:-1])
    elif ('-' in inp_str):
        vals = inp_str.split('-')
        return (float(vals[0])+float(vals[1]))/2.0
    else:
        return float(inp_str)

def generation_prediction_points(inp_df,continuous_cols,categorical_cols):
    prediction_df = inp_df.copy()

    prediction_df['passing_yards'] = prediction_df['receiving_yards']

    prediction_df['receive_touchdown'] = prediction_df['pass_touchdown']
    prediction_df['receive_touchdown_points'] = (
        prediction_df['receive_touchdown'].apply(lambda x: get_val_from_range(x)) * all_points_dict['receive_touchdown']
    ).astype(int)

    prediction_df['extra_point'] = (
        prediction_df['pass_touchdown'].apply(lambda x: get_val_from_range(x)) +
        prediction_df['rush_touchdown'].apply(lambda x: get_val_from_range(x))
    )
    prediction_df['extra_point_points'] = (
        prediction_df['extra_point'].apply(lambda x: get_val_from_range(x)) * kicker_points['extra_point']
    ).astype(int)


    prediction_df['close_field_goal_success'] = (
        prediction_df['close_field_goal_attempts'].apply(lambda x: get_val_from_range(x)) * prediction_df['close_field_goal_success_rate']
    ).astype(int)
    prediction_df['close_field_goal_success_points'] = (
        prediction_df['close_field_goal_success'] * all_points_dict['close_field_goal_success']
    ).astype(int)
    prediction_df['close_field_goal_failure'] = (
        prediction_df['close_field_goal_attempts'].apply(lambda x: get_val_from_range(x)) * (1.-prediction_df['close_field_goal_success_rate'])
    ).astype(int)
    prediction_df['close_field_goal_failure_points'] = (
        prediction_df['close_field_goal_failure'] * all_points_dict['close_field_goal_failure']
    ).astype(int)

    prediction_df['far_field_goal_success'] = (
        prediction_df['far_field_goal_attempts'].apply(lambda x: get_val_from_range(x)) * prediction_df['far_field_goal_success_rate']
    ).astype(int)
    prediction_df['far_field_goal_success_points'] = (
        prediction_df['far_field_goal_success'] * all_points_dict['far_field_goal_success']
    ).astype(int)
    prediction_df['far_field_goal_failure'] = (
        prediction_df['far_field_goal_attempts'].apply(lambda x: get_val_from_range(x)) * (1.-prediction_df['far_field_goal_success_rate'])
    ).astype(int)
    prediction_df['far_field_goal_failure_points'] = (
        prediction_df['far_field_goal_failure'] * all_points_dict['far_field_goal_failure']
    ).astype(int)

    prediction_df['defensive_points_allowed_points'] = prediction_df['defensive_points_allowed'].apply(
        lambda x: defensive_points_allowed(int(get_val_from_range(x)))
    )

    for col in prediction_df.columns.values:
        if ((col in all_points_dict) and (col!='defensive_points_allowed')):
            prediction_df[col+'_points'] = (
                prediction_df[col].apply(lambda x: get_val_from_range(x)*all_points_dict[col])
            ).astype(int)

    return prediction_df

def load_true_values_points(year_list,key_fields = ['season','week','team','opponent']):

    norm_data_path = normalize_raw_input.get_normalized_data_path()

    true_df = utilities.aggregate_read_data_files("weekly_team_data_season",norm_data_path,year_list)

    true_df['receive_touchdown'] = true_df['pass_touchdown']
    true_df['extra_point'] = true_df['extra_point_success']
    true_df['far_field_goal_success'] = (
        true_df['kick_distance_40_49_success'] + true_df['kick_distance_50_success']
    )
    true_df['far_field_goal_failure'] = (
        true_df['kick_distance_40_49_fail'] + true_df['kick_distance_50_fail']
    )
    true_df['close_field_goal_success'] = true_df['kick_distance_0_39_success']
    true_df['close_field_goal_failure'] = true_df['kick_distance_0_39_fail']
    true_df['defensive_points_allowed'] = true_df['opponent_points']
    true_df['offensive_interception'] = true_df['interception']

    def_reverse_df = true_df[key_fields+['fumble_forced','interception']].rename(columns={
        'team':'opponent',
        'opponent':'team',
        'interception':'defensive_interception',
        'fumble_forced':'defensive_fumble_forced'
    })
    true_df = true_df.merge(def_reverse_df,on=key_fields)

    for col in true_df.columns.values:
        if ( (col in all_points_dict) and (col != 'defensive_points_allowed') ):
            true_df[col+'_points'] = (true_df[col].apply(lambda x: x*all_points_dict[col])).astype(int)

    true_df['defensive_points_allowed'] = true_df['opponent_points']
    true_df['defensive_points_allowed_points'] = true_df['defensive_points_allowed'].apply(
        lambda x: defensive_points_allowed(int(x))
    )

    return true_df


def gen_top_team_list(inp_df,inp_cols,n_teams,key_fields=['season','week','team','opponent']):
    if isinstance(inp_cols,str):
        inp_cols = [inp_cols]
    if isinstance(n_teams,int):
        n_teams = [n_teams]

    lookback_list = n_teams
    season_week_col_lookback_team_top_lists = {}
    for season in inp_df['season'].unique():
        season_week_col_lookback_team_top_lists[season] = {}
        for week in inp_df['week'].unique():
            season_week_col_lookback_team_top_lists[season][week] = {}
            for col in inp_cols:
                season_week_col_lookback_team_top_lists[season][week][col] = {}
                for lookback in lookback_list:
                    season_week_col_lookback_team_top_lists[season][week][col][lookback] = []

    for col in inp_cols:
        current_df = inp_df[key_fields+[col]].copy()
        current_df['dummy'] = 0
        best_performance_gb_pre = (
            current_df[['season','week',col,'dummy']].drop_duplicates().groupby(
                ['season','week']
            )
        )
        for lookback in lookback_list:
            best_performance_gb = best_performance_gb_pre.apply(lambda grp: grp.nlargest(lookback,col))[col]
            best_performance_df = best_performance_gb.reset_index(drop=False)
            top_performing_team_df = current_df.merge(
                best_performance_df,on=['season','week',col]
            )

            for ind,row in top_performing_team_df.iterrows():
                season = row['season']
                week   = row['week']
                team   = row['team']
                season_week_col_lookback_team_top_lists[season][week][col][lookback].append(team)
    return season_week_col_lookback_team_top_lists

# Only start and end year, can read from default max range
def __read_args__():

    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

    parser.add_argument('--prediction_file_name', type=str, required=True,
        help='Prediction file name to load/use'
    )

    parser.add_argument('--top_stats_list', nargs='+', required=True,
        help='Prediction file name to load/use'
    )

    parser.add_argument('--model_version', type=str, nargs='?',
        default=model_generation.__model_version__,
        help='The version to use for models'
    )


    args = parser.parse_args()

    input_arguments = vars(args)

    argument_validation.run_argument_validation(input_arguments)

    return input_arguments


def analyze(input_arguments):

    inp_dict = load_predictions(input_arguments)

    inp_df = inp_dict['output_result']

    print("Loading predictions...")
    prediction_df = generation_prediction_points(
        inp_df,
        inp_dict['continuous_cols'],
        inp_dict['propogate_cols']
    )

    key_fields = ['season','week','team','opponent']

    points_cols = []
    values_cols = []
    rename_cols = {}
    for col in prediction_df.columns.values:
        if ( col.endswith('_points')  ):
            rename_cols[col] = col.replace('_predicted','')
            points_cols.append(rename_cols[col])
            if (col=='defensive_points_allowed_points'):
                values_cols.append('defensive_points_allowed')
            else:
                values_cols.append(col.replace('_predicted','').replace('_points',''))

    points_df = prediction_df.rename(columns=rename_cols)[key_fields+points_cols].copy()
    points_df['total_points'] = points_df[points_cols].sum(axis=1)

    values_df = prediction_df.rename(columns=rename_cols)[key_fields+values_cols].copy()

    print("Loading true values...")

    year_list=[]
    for year in values_df['season'].unique():
        year_list.append(year)
    true_df = load_true_values_points(year_list).merge(points_df[key_fields],on=key_fields)
    true_points_df = true_df[key_fields+points_cols].copy()
    true_points_df['total_points'] = true_points_df[points_cols].sum(axis=1)
    true_values_df = true_df[key_fields+values_cols].copy()

    points_cols.append('total_points')

    print("Plotting true/pred...")
    plot_pred_true(values_cols,values_df,true_values_df,input_arguments)
    plot_pred_true(points_cols,points_df,true_points_df,input_arguments)

    true_values_df_range_replaced = true_values_df.copy()
    class_values_ranges = inp_dict['discreet_cols']
    for key in class_values_ranges:
        if (key in true_values_df_range_replaced.columns):
            true_values_df_range_replaced = model_generation.gen_field_ranges(
                true_values_df_range_replaced,
                key,
                class_values_ranges[key]
            )

    plot_confusion(values_cols,values_df,true_values_df_range_replaced,input_arguments)


    print("Generating top performing teams...")
    n_lookback = input_arguments['top_stats_list']
    true_top_teams = gen_top_team_list(true_points_df,points_cols,n_lookback)
    pred_top_teams = gen_top_team_list(     points_df,points_cols,n_lookback)

    n_teams = true_points_df['team'].unique().shape[0]
    output_str = "Random guessing: \n"
    output_str+= "\tGet top: {:5.2f}%\n".format(1*100.0/true_points_df['team'].unique().shape[0])
    for n in n_lookback:
        if (n==1):
            continue
        output_str+= "\tGet any top {:2d}: {:5.2f}%\tGet exact top {:2d}: {:5.2e}%\n".format(
            n,
            n*100./n_teams,
            n,
            100./nCk(n_teams,n)
        )

    for col in points_cols+['total_points']:
        output_str+= col+'\n'
        for n in n_lookback:
            n_true = 0.0
            n_pred_match = 0.0

            n_pred = 0.0
            n_true_match = 0.0
            many_at_top=False
            for year in true_points_df['season'].unique():
                for week in true_points_df['week'].unique():
                    true_list = true_top_teams[year][week][col][n]
                    pred_list = pred_top_teams[year][week][col][n]
                    if (len(true_list)>0.5*n_teams):
                        many_at_top=True
                    n_true += len(true_list)
                    for i in range(0,len(pred_list)):
                        if (pred_list[i] in true_list):
                            n_pred_match += 1.0

                    n_pred += len(pred_list)
                    for i in range(0,len(true_list)):
                        if (true_list[i] in pred_list):
                            n_true_match += 1.0

            many_warning = "\n"
            if (many_at_top):
                many_warning = "    Warning: Field trends to low values\n"
            output_str+= "\tTop {:1d}:    n_true_at_level={:6d},    n_pred_that_match={:6d},    recall   ={:8.5f}{:s}".format(
                int(n),
                int(n_true),
                int(n_pred_match),
                n_pred_match/n_true,
                many_warning
            )
            output_str+= "\tTop {:1d}:    n_pred_at_level={:6d},    n_true_that_match={:6d},    precision={:8.5f}{:s}".format(
                int(n),
                int(n_pred),
                int(n_true_match),
                n_true_match/n_pred,
                many_warning
            )
    print(output_str)

    txt_output_path = get_analyze_path(input_arguments["analyze_version"]) + input_arguments['prediction_file_name'] + "/"
    os.makedirs(txt_output_path,exist_ok=True)
    txt_full_fn = txt_output_path+input_arguments['prediction_file_name']+"_top_stats.txt"
    with open(txt_full_fn,'w') as f:
        f.write(output_str)


if __name__ == "__main__":
    inp_args = __read_args__()
    analyze(inp_args)
