import sys

#from util.utilities import PCACols,ZScaler
from sklearn.decomposition import PCA

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import pickle as pkl
import numpy as np
import pandas as pd
import os
import argparse
import datetime

#TODO: proper implementation
sys.path.append('/home/sean/Documents/Fantasy_Football/')

from normalize import normalize_raw_input
from util import utilities
from util import argument_validation
from model.data_structures import ZScaler,PCACols,ModelWrapper


__model_version__='0.1.0'

def __read_args__():

    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

    parser.add_argument('--normalization_version', type=str, nargs='?',
        default=normalize_raw_input.__normalization_version__,
        help='The version to use for normalization'
    )
    parser.add_argument('--norm_start_year', type=int, nargs='?', default=1999,
        help='The starting year for input data (Earliest available data is from 1999)'
    )
    parser.add_argument('--norm_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for input data (Latest is current year)'
    )

    parser.add_argument('--process_start_year', type=int, nargs='?', default=1999,
        help='The starting year for input data (Earliest available data is from 1999)'
    )
    parser.add_argument('--process_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for input data (Latest is current year)'
    )

    parser.add_argument('--predict_start_year', type=int, nargs='?', default=1999,
        help='The starting year for input data (Earliest available data is from 1999)'
    )
    parser.add_argument('--predict_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for input data (Latest is current year)'
    )

    parser.add_argument('--n_rolling', type=int, nargs='?',default=3,
        help='The number of months for lookback'
    )
    parser.add_argument('--n_components_team', type=int, nargs='?',default=24,
        help='The number of PCA components for the team data'
    )
    parser.add_argument('--n_components_opp', type=int, nargs='?',default=15,
        help='The number of PCA components for the opposing team data'
    )

    parser.add_argument('--input_scaler_file_name', type=str, nargs='?',
        help='Optional scaler file to load/use'
    )

    parser.add_argument('--reg_model_type', type=str, nargs='?',
        default='Linear',
        help='The ML model to use for regression [Linear,SVM,Forest,MLP]'
    )
    parser.add_argument('--clf_model_type', type=str, nargs='?',
        default='Logistic',
        help='The ML model to use for classification [Logistic,SVM,Forest,MLP]'
    )
    
    parser.add_argument('--output_scaler_file_name', type=str, nargs='?',
        help='Optional scaler file to save, otherwise uses date'
    )

    parser.add_argument('--model_version', type=str, nargs='?',
        default=__model_version__,
        help='The version to use for models'
    )

    parser.add_argument('--input_models_file_name', type=str, nargs='?',
        help='Optional ML file to load/use'
    )

    parser.add_argument('--output_models_file_name', type=str, nargs='?',
        help='Optional ML file to save, othwerwise uses date'
    )

    parser.add_argument('--predict_values', action='store_true',
        help='Whether to forecast values for predict years or not'
    )

    parser.add_argument('--prediction_file_name', type=str, nargs='?',
        help='Optional save data with prediction'
    )

    args = parser.parse_args()
    inp_args = vars(args)

    inp_args['generate_model'] = inp_args['input_models_file_name'] is None

    argument_validation.run_argument_validation(inp_args)

    return inp_args

def generate_df_rolling_means(
    input_df,
    n_lookback,
    fields_to_rollup,
    save_fields = ['season','week','team'],
    key_field = 'team',
):
    output_df = input_df[save_fields].copy()
    for col in fields_to_rollup:
        new_col = col+"_"+str(n_lookback)
        output_df[new_col] = 0.0
        for team in input_df[key_field].unique():
            roll_ind = output_df[key_field]==team
            output_df.loc[roll_ind,new_col] = input_df.loc[roll_ind,col].rolling(n_lookback).mean()
    return output_df

def gen_field_ranges( inp_df, col, range_starts ):
    assert isinstance(range_starts,list) # Need a list input
    assert len(range_starts)>1 # Only operate when there are a number of values

    out_df = inp_df.copy()

    for i in range(0,len(range_starts)-1):
        start = range_starts[i]
        end   = range_starts[i+1] - 1
        if (start==end):
            new_col = col + "_range__" + str(start)
            out_df[new_col] = 0
            out_df.loc[ out_df[col].astype(int) == start,new_col] = 1
        else:
            new_col = col + "_range_" + str(start) + "_" + str(end)
            out_df[new_col] = 0
            out_df.loc[
                (out_df[col].astype(int) >= start) &
                (out_df[col].astype(int) <= end  ),
                new_col
            ] = 1

    end = range_starts[-1]
    new_col = col + "_range_" + str(end) + "_"
    out_df[new_col] = 0
    out_df.loc[ out_df[col].astype(int) == end,new_col] = 1

    return out_df

# Generate all the features, including using opposing team stats
def generate_weekly_team_features_values( inp_df, key_fields, n_rolling ):
    norm_weekly_df = inp_df.copy()

    opposing_turnovers_to_reverse = norm_weekly_df[
        key_fields + ['fumble_forced','interception','points']
    ].rename(columns={
        'interception': 'defensive_interception',
        'fumble_forced': 'defensive_fumble_forced',
        'points': 'defensive_points_allowed',
        'team': 'opponent',
        'opponent': 'team',
    })
    norm_weekly_df = norm_weekly_df.merge(
        opposing_turnovers_to_reverse,
        on=key_fields
    ).rename(
        columns={
        'interception': 'offensive_interception',
        'fumble_forced': 'offensive_fumble_forced',
    })


    weekly_df = norm_weekly_df.copy()
    weekly_df['pass_td_ratio'] = weekly_df['pass_touchdown'] / (weekly_df['touchdown']+1e-7)
    weekly_df['extra_point_success_rate'] = weekly_df['extra_point_success'] / (
        weekly_df['extra_point_attempt']+1e-7
    )

    weekly_df['close_field_goal_success'] = weekly_df['kick_distance_0_39_success']
    weekly_df['close_field_goal_fail'] = weekly_df['kick_distance_0_39_fail']
    weekly_df['close_field_goal_attempts'] = (
        weekly_df['close_field_goal_success'] + weekly_df['close_field_goal_fail']
    )
    weekly_df['close_field_goal_success_rate'] = weekly_df['close_field_goal_success'] / (
        weekly_df['close_field_goal_attempts'] + 1e-7
    )

    weekly_df['far_field_goal_success'] = weekly_df['kick_distance_40_49_success'] + weekly_df['kick_distance_50_success']
    weekly_df['far_field_goal_fail'] = weekly_df['kick_distance_40_49_fail'] + weekly_df['kick_distance_50_fail']
    weekly_df['far_field_goal_attempts'] = (
        weekly_df['far_field_goal_success'] + weekly_df['far_field_goal_fail']
    )
    weekly_df['far_field_goal_success_rate'] = weekly_df['far_field_goal_success'] / (
        weekly_df['far_field_goal_attempts'] + 1e-7
    )

    # Zero attempts for field goals don't leave much to interpret
    for col_pre in ['close_field_goal','far_field_goal']:
        col = col_pre + '_success_rate'
        weekly_df['old_'+col] = weekly_df[col]
        for team in weekly_df['team'].unique():
            team_df = weekly_df.loc[weekly_df['team']==team].sort_values(['season','week'])
            prev_rate = team_df[col_pre+'_success'].sum()/team_df[col_pre+'_attempts'].sum()
            for idx,row in team_df.iterrows():
                if ( row[col_pre+'_attempts']<1 ):
                    weekly_df.loc[idx,col] = prev_rate
                else:
                    prev_rate = row[col]

    weekly_df['pass_play_rate'] = weekly_df['pass_attempt'] / (
        weekly_df['pass_attempt'] + weekly_df['rush_attempt'] + 1e-7
    )
    weekly_df['pass_complete_rate'] = weekly_df['complete_pass'] / (
        weekly_df['pass_attempt'] + 1e-7
    )
    weekly_df['average_pass_yards'] = weekly_df['receiving_yards'] / (
        weekly_df['complete_pass'] + 1e-7
    )
    weekly_df['average_rush_yards'] = weekly_df['rushing_yards'] / (
        weekly_df['rush_attempt'] + 1e-7
    )

    weekly_df['fumble_recovery_rate'] = weekly_df['recovery'] / ( weekly_df['fumble'] + 1e-7 )


    team_fields = [
        'rushing_yards', 'passing_yards', 'receiving_yards',
        'pass_play_rate', 'pass_complete_rate',
        'average_pass_yards', 'average_rush_yards',
        'passing_yards_40', 'rushing_recieving_yards_40',

        'pass_touchdown', 'rush_touchdown', 'td_yards_40',
        'extra_point_success_rate',
        'close_field_goal_attempts', 'close_field_goal_success_rate',
        'far_field_goal_attempts', 'far_field_goal_success_rate',

        'offensive_interception',
        'offensive_fumble_forced', 'fumble_not_forced', 'recovery',
        'fumble_recovery_rate',

        'penalty_yards',

        'assist_tackle', 'qb_hit', 'solo_tackle', 'sack',

        'defensive_points_allowed',
        'defensive_fumble_forced',
        'defensive_interception',
    ]
    team_weekly_fields_df = weekly_df[key_fields+team_fields].copy()


    rolling_weekly_df = generate_df_rolling_means(
        team_weekly_fields_df,
        n_rolling,
        team_fields,
        save_fields = ['season','week','team','opponent'],
    )

    rename_dict = {'team':'opponent','opponent':'team'}
    opp_fields = []
    opp_fields_to_use = [
        'rushing_yards', 'receiving_yards',
        'pass_play_rate', 'pass_complete_rate',
        'passing_yards_40', 'rushing_recieving_yards_40',

        'pass_touchdown', 'rush_touchdown',
        'extra_point_success_rate',
        'close_field_goal_success_rate',
        'far_field_goal_success_rate',

        'qb_hit', 'sack',

        'offensive_interception',
        'offensive_fumble_forced',

        'defensive_points_allowed',
        'defensive_fumble_forced',
        'defensive_interception',
    ]
    for field in opp_fields_to_use:
        field_x = field + "_" + str(n_rolling)
        rename_dict[field_x] = 'opponent_' + field_x
        opp_fields.append(rename_dict[field_x])
        opposing_team_df = rolling_weekly_df.rename(columns=rename_dict)[key_fields+opp_fields].copy()

    team_fields_to_scale = []
    for field in team_fields:
        team_fields_to_scale.append( field + '_' + str(n_rolling) )


    #TODO: move
    forecast_values_for_team = [
        'rushing_yards',
        'receiving_yards',
        'passing_yards',
        'complete_pass',
        'touchdown',
        'pass_touchdown',
        'rush_touchdown',
        'td_yards_40',
        'extra_point_success',
        'close_field_goal_attempts',
        'close_field_goal_success_rate',
        'far_field_goal_attempts',
        'far_field_goal_success_rate',
        'sack',
        'fumble_recovery_rate',
        'fumble',
        'offensive_interception',

        'defensive_points_allowed', # scores for D
        'defensive_fumble_forced', # scores for D
        'defensive_interception', # scores for D
    ]
    values_to_predict_df = weekly_df[key_fields+forecast_values_for_team].copy()


    return {
        'value_df':values_to_predict_df,
        'team_df':rolling_weekly_df,
        'opposing_df':opposing_team_df,
        'team_fields':team_fields_to_scale,
        'opposing_fields':opp_fields,
    }

#TODO: implement
def load_scale_model():
    pass

def gen_scale_model( inp_dict ):
    scaler = ZScaler( inp_dict['team_df'],columns=inp_dict['team_fields'])
    scaler.add( inp_dict['opposing_df'][inp_dict['opposing_fields']] )
    return scaler


def scale_combine_team_opposition( inp_dict, scaler, key_fields ):
    rolling_scaled_weekly_df = scaler.scale_cols( inp_dict['team_df'], inp_dict['team_fields']).dropna()
    opposing_scaled_weekly_df = scaler.scale_cols( inp_dict['opposing_df'], inp_dict['opposing_fields']).dropna()

    joined_weekly_df = rolling_scaled_weekly_df.merge(
        opposing_scaled_weekly_df,
        on=['season','week','team','opponent']
    )

    return joined_weekly_df

#TODO:implement
def load_pca_model():
    pass

def gen_pca_model( inp_df, inp_fields, n_components ):
    pca_model = PCACols(
        inp_df,
        inp_fields,
        n_components
    )
    return pca_model


def generate_models_from_list(
        fields_to_model,
        feature_df =None,
        value_df = None,
        reuse_model_wrapper = None,
        model = None,
        cv_parameters = None,
        classifier = False,
        use_cols=None,
        scoring=None,
        test_size=0.20,
        n_jobs=1,
        cv=3,
        balance_sample=None
):
    assert ((feature_df is not None) and (value_df is not None)) or ( reuse_model_wrapper is not None )

    if (isinstance(fields_to_model,str)):
        fields_to_model=[fields_to_model]
    assert isinstance(fields_to_model,list) # input_fields must be list

    # Need to verify model, parameters are either singular, or match the dimension of fields
    if ( model is None ):
        if ( classifier ):
            model=LogisticRegression()
        else:
            model=LinearRegression()

    model_list = []
    if ( isinstance(model,list) ):
        assert len(model)==len(fields_to_model) # Must have same input lens
        model_list=model
    else:
        for i in range(0,len(fields_to_model)):
            model_list.append(model)

    use_col_list = []
    if (isinstance(use_cols,list) and (len(use_cols)>0) ):
        if (isinstance(use_cols[0],list)):
            assert len(use_cols)==len(fields_to_model) # Must have same input lens
        use_col_list = use_cols
    else:
        for i in range(0,len(fields_to_model)):
            use_col_list.append(use_cols)

    scoring_list = []
    if (isinstance(scoring,list)):
        assert len(scoring)==len(fields_to_model) # Must have same input lens
        scoring_list = scoring
    else:
        for i in range(0,len(fields_to_model)):
            scoring_list.append(scoring)

    cv_param_list = []
    if (isinstance(cv_parameters,list)):
        assert len(cv_parameters)==len(fields_to_model) # Must have same input lens
        cv_param_list = cv_parameters
    else:
        for i in range(0,len(fields_to_model)):
            cv_param_list.append(cv_parameters)

    my_models = 0
    if (reuse_model_wrapper is None):
        my_models = ModelWrapper( feature_df, value_df )
    else:
        my_models = reuse_model_wrapper

    for i in range(0,len(fields_to_model)):
        my_models.train_model(
            fields_to_model[i],
            model_list[i],
            parameters = cv_param_list[i],
            use_cols = use_col_list[i],
            scoring=scoring_list[i],
            test_size=test_size,
            n_jobs=n_jobs,
            cv=cv,
            balance_sample=balance_sample,
        )
    return my_models


def get_model_path(version=__model_version__):
    return utilities.get_project_dir()+'data/model/'+version+'/'

def get_prediction_path(version=__model_version__):
    return utilities.get_project_dir()+'data/predicted_values/'+version+'/'

def get_features_values_dict(input_arguments):

    input_year_list = utilities.gen_year_list(
        input_arguments['norm_start_year'],
        input_arguments['norm_end_year']
    )

    norm_weekly_df = utilities.aggregate_read_data_files(
        "weekly_team_data_season",
        normalize_raw_input.get_normalized_data_path(input_arguments['normalization_version']),
        input_year_list
    )

    key_fields = ['season','week','team','opponent']

    return generate_weekly_team_features_values( norm_weekly_df, key_fields, input_arguments['n_rolling'] )


def create_model(input_arguments,output_dfs,key_fields=['season','week','team','opponent']):

    # Zscale fields
    if (input_arguments['input_scaler_file_name'] is None):
        scaler = gen_scale_model(output_dfs)

        if (input_arguments['output_scaler_file_name'] is not None):
            output_name = get_model_path() + input_arguments['output_scaler_file_name']
        else:
            output_name = get_model_path() + \
                "scaler_"+\
                str(input_arguments["process_start_year"])+\
                "_"+\
                str(input_arguments["process_end_year"])+\
                ".pkl"
        os.makedirs(get_model_path(),exist_ok=True)
        with open(output_name,'wb') as f:
            pkl.dump(scaler,f)
        print("Wrote "+output_name)
    else:
        input_name = get_model_path()+input_arguments['input_scaler_file_name']
        print("Using file "+input_name)
        with open(input_name, 'rb') as f:
            scaler = pkl.load(f)

    #TODO:Consider inplementing Reduce fields
    #TODO:conditional load
    #team_pca = gen_pca_model(
    #    scaled_dict['joined_weekly_df'],
    #    scaled_dict['team_fields'],
    #    input_arguments['n_components_team']
    #)
    #transformed_team = team_pca.transform( scaled_dict['joined_weekly_df'] )
    #opp_pca = gen_pca_model(
    #    scaled_dict['joined_weekly_df'],
    #    scaled_dict['opposing_fields'],
    #    input_arguments['n_components_opp']
    #)
    #transformed_opp = opp_pca.transform( scaled_dict['joined_weekly_df'] )
    #feature_array = np.concatenate([transformed_team,transformed_opp],axis=1)


    values_cols = [
        'rushing_yards', 'receiving_yards', 'passing_yards',
        'complete_pass',
        'touchdown', 'pass_touchdown', 'rush_touchdown', 'td_yards_40',
        'extra_point_success',
        'close_field_goal_attempts', 'close_field_goal_success_rate',
        'far_field_goal_attempts', 'far_field_goal_success_rate',
        'sack',
        'fumble', 'fumble_recovery_rate',
        'offensive_interception',
        'defensive_points_allowed',
        'defensive_fumble_forced', 'defensive_interception'
    ]

    continuous_values_cols = [
        #'rushing_yards', 'receiving_yards',
        'rushing_yards', 'receiving_yards', 'complete_pass',
    ]

    propogate_cols = [
        'close_field_goal_success_rate',
        'far_field_goal_success_rate',
        'fumble_recovery_rate',
    ]

    class_values_ranges = {
        'touchdown':[0,1,2,3,4,5],
        'pass_touchdown':[0,1,2,3],
        'rush_touchdown':[0,1,2,3],
        'td_yards_40':[0,1],
        'close_field_goal_attempts':[0,1,2,3,4],
        'far_field_goal_attempts':[0,1,2,3],
        'sack':[0,2,4,6],
        'fumble':[0,1,2],
        'offensive_interception':[0,1,2],
        'defensive_points_allowed':[0,1,7,14,21,28,35],
        'defensive_fumble_forced':[0,2,4],
        'defensive_interception':[0,1,2],
    }


    team_fields         = output_dfs['team_fields']
    opposing_fields     = output_dfs['opposing_fields']

    scaled_joined_df_pre = scale_combine_team_opposition( output_dfs, scaler, key_fields )
    scaled_joined_df    = utilities.filter_df_year( scaled_joined_df_pre     , input_arguments['process_start_year'], input_arguments['process_end_year'] )
    values_df           = utilities.filter_df_year( output_dfs['value_df']   , input_arguments['process_start_year'], input_arguments['process_end_year'] )
    team_rolling_df     = utilities.filter_df_year( output_dfs['team_df']    , input_arguments['process_start_year'], input_arguments['process_end_year'] )
    opposing_rolling_df = utilities.filter_df_year( output_dfs['opposing_df'], input_arguments['process_start_year'], input_arguments['process_end_year'] )

    joined_df = team_rolling_df[key_fields+team_fields].merge(
        opposing_rolling_df[key_fields+opposing_fields],
        on=key_fields
    )

    # For NN models
    n_features = len(team_fields)
    n_   = n_features
    n_23 = int(n_features*2./3)
    n_2  = int(n_features/2.)
    n_3  = int(n_features/3.)

    regressor_model_dict = {
        'Linear': LinearRegression(),
        'Forest':RandomForestRegressor(),
        'MLP':MLPRegressor(
            activation='identity', # This is consistently best
            solver='adam',
            max_iter=1000,
        ),
        'SVM':SVR(),
    }

    classifier_model_dict = {
        'Logistic': LogisticRegression(),
        'Forest':RandomForestClassifier(),
        'MLP':MLPClassifier(
            activation='identity', # This is consistently best
            solver='adam',
            max_iter=1000,
        ),
        'SVM':SVC(),
    }
    
    regressor_cv_param_dict = {
        'Linear':{
            'fit_intercept':[True,False],
        },
        'Forest':{
            'n_estimators':[10,50,100,300],
            'max_depth':[None,10,15],
            'max_features':[None,'sqrt'],
            'min_samples_leaf':[1,0.1,0.05,0.01],
        },
        'MLP':{
            'hidden_layer_sizes': [
                (n_  ,),
                (n_23,),
                (n_2 ,),
                (n_3 ,),
                (n_  , n_23,),
                (n_23, n_2 ,),
                (n_2 , n_3 ,),
                (n_23, n_  ,),
                (n_2 , n_23,),
                (n_3 , n_2 ,),
            ],
        },
        'SVM':{
            'C':10.**(np.arange(-3, 2, 1.0)),
            'kernel':['poly','rbg','sigmoid'],
            'gamma':['scale','auto'],
        }
    }

    classifier_cv_param_dict = {
        'Logistic':{
            'fit_intercept':[True,False],
            'C':10.**(np.arange(-3, 2, 0.5))
        },
        'Forest':{
            'n_estimators':[10,50,100,300],
            'max_depth':[None,10,15],
            'max_features':[None,'sqrt'],
            'min_samples_leaf':[1,0.1,0.05,0.01],
        },
        'MLP':{
            'hidden_layer_sizes': [
                (n_  ,),
                (n_23,),
                (n_2 ,),
                (n_3 ,),
                (n_  , n_23,),
                (n_23, n_2 ,),
                (n_2 , n_3 ,),
                (n_23, n_  ,),
                (n_2 , n_23,),
                (n_3 , n_2 ,),
            ],
        },
        'SVM':{
            'C':10.**(np.arange(-3, 2, 1.0)),
            'kernel':['poly','rbg','sigmoid'],
            'gamma':['scale','auto'],
        },
    }

    this_reg_model = regressor_model_dict    [input_arguments['reg_model_type']]
    this_reg_param = regressor_cv_param_dict [input_arguments['reg_model_type']]
    this_clf_model = classifier_model_dict   [input_arguments['clf_model_type']]
    this_clf_param = classifier_cv_param_dict[input_arguments['clf_model_type']]

    class_cols = {}
    class_values_list = []
    for key in class_values_ranges:
        values_df = gen_field_ranges(values_df,key,class_values_ranges[key])

        for col in values_df:
            if ( (key in col) and ("_range_" in col) ):
                if (key in class_cols):
                    class_cols[key].append(col)
                else:
                    class_cols[key] = [col]
                class_values_list.append(col)

    reg_models = generate_models_from_list(
        fields_to_model=continuous_values_cols,
        feature_df = joined_df,
        value_df = values_df,
        model = this_reg_model,
        cv_parameters = this_reg_param,
        test_size=0.20,
        n_jobs=4,
        cv=3,
    )
    print(reg_models.model(continuous_values_cols[0]))

    class_models = generate_models_from_list(
        fields_to_model=class_values_list,
        feature_df = scaled_joined_df,
        value_df = values_df,
        model = this_clf_model,
        cv_parameters = this_clf_param,
        test_size=0.20,
        n_jobs=4,
        scoring='recall',
        cv=3,
        balance_sample=1.2,
    )

    combined_models = {
        'reg_models': reg_models,
        'class_models': class_models,
        'scaler': scaler,
        'propogate_cols': propogate_cols,
    }

    if (input_arguments['output_models_file_name'] is not None):
        output_name = get_model_path(input_arguments['model_version']) + input_arguments['output_models_file_name']
    else:
        output_name = get_model_path(input_arguments['model_version']) + \
            "models_"+\
            str(input_arguments["process_start_year"])+\
            "_"+\
            str(input_arguments["process_end_year"])+\
            ".pkl"
    os.makedirs(get_model_path(input_arguments['model_version']),exist_ok=True)
    with open(output_name, 'wb') as f:
        pkl.dump(combined_models,f)
        print("Wrote file "+output_name)

    return combined_models


def predict(input_arguments,output_dfs,combined_models,key_fields=['season','week','team','opponent']):

    scaled_joined_df = scale_combine_team_opposition( output_dfs, combined_models['scaler'], key_fields )

    team_fields         = output_dfs['team_fields']
    opposing_fields     = output_dfs['opposing_fields']

    values_df           = output_dfs['value_df']
    team_rolling_df     = output_dfs['team_df']
    opposing_rolling_df = output_dfs['opposing_df']


    joined_df = team_rolling_df[key_fields+team_fields].merge(
        opposing_rolling_df[key_fields+opposing_fields],
        on=key_fields
    )

    scaled_joined_out_df = utilities.filter_df_year(
        scaled_joined_df,
        input_arguments['predict_start_year'],
        input_arguments['predict_end_year']
    ).dropna()

    joined_out_df = utilities.filter_df_year(
        joined_df,
        input_arguments['predict_start_year'],
        input_arguments['predict_end_year']
    )

    values_out_df = utilities.filter_df_year(
        values_df,
        input_arguments['predict_start_year'],
        input_arguments['predict_end_year']
    )

    output_dict = {
        'propogate_df':values_out_df[key_fields+combined_models['propogate_cols']],
    }

    for col in combined_models['reg_models'].get_model_predicted_fields():
        output_dict[col] = combined_models['reg_models'].predict(col,joined_out_df)

    for col in combined_models['class_models'].get_model_predicted_fields():
        output_dict[col] = combined_models['class_models'].predict_proba(col,scaled_joined_out_df)

    os.makedirs(get_prediction_path(input_arguments['model_version']),exist_ok=True)
    output_name = get_prediction_path(input_arguments['model_version']) + input_arguments['prediction_file_name']
    with open(output_name, 'wb') as f:
        pkl.dump(output_dict,f)
    print("Wrote "+output_name)


def run_model_gen_prediction(inp_args):
    output_dfs = get_features_values_dict(inp_args)

    ml_model_dict = {}
    if (inp_args['input_models_file_name'] is None):
        ml_model_dict = create_model(inp_args,output_dfs)
    else:
        input_name = get_model_path()+inp_args['input_models_file_name']
        print("Reading "+input_name+"...")
        with open(input_name,'rb') as f:
            ml_model_dict = pkl.load(f)

    if (inp_args['predict_values']):
        predict(inp_args,output_dfs,ml_model_dict)

if __name__ == "__main__":
    inp_args = __read_args__()
    run_model_gen_prediction(inp_args)
