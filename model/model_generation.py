import sys

from sklearn.decomposition import PCA

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
from model.data_structures import ZScaler,PCACols,ModelWrapper,get_index_resampled_categories_reduce_largest,MLTrainingHelper,PCATeamOpp


__model_version__='0.1.0'

def __read_args__():

    return argument_validation.build_args([
        'normalization_version',
        'norm_start_year',
        'norm_end_year',
        'generate_model',
        'model_version',
        'input_models_file_name',
        'output_models_file_name',
        'process_start_year',
        'process_end_year',
        'reg_model_type',
        'clf_model_type',
        'n_rolling',
        'n_components_team',
        'n_components_opp',
        'predict_values',
        'prediction_file_name',
        'predict_start_year',
        'predict_end_year',
    ])

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

    out_dict = {}

    for i in range(0,len(range_starts)-1):
        start = range_starts[i]
        end   = range_starts[i+1] - 1
        val   = ""
        if (start==end):
            val = str(start)
        else:
            val = str(start) + "-" + str(end)
        for j in range(start,end+1):
            out_dict[j] = val

    end = range_starts[-1]
    out_dict[end] = str(end)+"+"
    out_df = inp_df.copy()

    out_df[col] = out_df[col].apply(lambda x: out_dict[x] if x in out_dict else out_dict[end])

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


def generate_models_from_list(
        fields_to_model,
        feature_df =None,
        value_df = None,
        scaler = None,
        pca_obj = None,
        reuse_model_wrapper = None,
        model = None,
        cv_parameters = None,
        classifier = False,
        use_cols=None,
        scoring=None,
        test_size=0.20,
        n_jobs=1,
        cv=3,
        balance_sample=None,
        multiclass=False,
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
        my_models = ModelWrapper( feature_df, value_df, scaler, pca_obj )
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
            multiclass=multiclass,
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
    scaler = gen_scale_model(output_dfs)

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
        'rushing_yards', 'receiving_yards', 'complete_pass',
    ]

    propogate_cols = [
        'close_field_goal_success_rate',
        'far_field_goal_success_rate',
        'fumble_recovery_rate',
    ]

    team_fields         = output_dfs['team_fields']
    opposing_fields     = output_dfs['opposing_fields']

    scaled_joined_df_pre = scale_combine_team_opposition( output_dfs, scaler, key_fields )
    scaled_joined_df    = utilities.filter_df_year( scaled_joined_df_pre     , input_arguments['process_start_year'], input_arguments['process_end_year'] ).dropna()
    team_rolling_df     = utilities.filter_df_year( output_dfs['team_df']    , input_arguments['process_start_year'], input_arguments['process_end_year'] ).dropna()
    opposing_rolling_df = utilities.filter_df_year( output_dfs['opposing_df'], input_arguments['process_start_year'], input_arguments['process_end_year'] ).dropna()
    values_df           = utilities.filter_df_year( output_dfs['value_df']   , input_arguments['process_start_year'], input_arguments['process_end_year'] )

    joined_df = team_rolling_df[key_fields+team_fields].merge(
        opposing_rolling_df[key_fields+opposing_fields],
        on=key_fields
    )

    scaled_joined_df = scaled_joined_df.merge(
        team_rolling_df[key_fields],
        on=key_fields
    )

    values_df = values_df.merge(
        team_rolling_df[key_fields],
        on=key_fields
    )

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
    class_values_list = []
    for key in class_values_ranges:
        values_df = gen_field_ranges(values_df,key,class_values_ranges[key])
        class_values_list.append(key)

    pca_comb = PCATeamOpp(
        scaler,
        team_rolling_df,
        team_rolling_df.drop(columns=key_fields).columns,
        input_arguments['n_components_team'],
        opposing_rolling_df,
        opposing_rolling_df.drop(columns=key_fields).columns,
        input_arguments['n_components_opp'],
    )
    team_pca_inp_df = team_rolling_df[key_fields+team_fields].merge(
        opposing_rolling_df[key_fields],
        on=key_fields
    ).copy().sort_values(key_fields)
    opp_pca_inp_df = team_rolling_df[key_fields].merge(
        opposing_rolling_df[key_fields+opposing_fields],
        on=key_fields
    ).copy().sort_values(key_fields)
    pca_data = pca_comb.transform( team_pca_inp_df, opp_pca_inp_df )
    pca_data_cols = [str(i) for i in range(0,pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data,columns = pca_data_cols)


    ml_helper = MLTrainingHelper(
        joined_df,
        scaled_joined_df,
        pca_df,
        input_arguments['reg_model_type'],
        input_arguments['clf_model_type'],
    )

    this_reg_model = ml_helper.get_regressor_model()
    this_reg_param = ml_helper.get_regressor_model_cv_params()
    this_reg_x     = ml_helper.get_regressor_input_x().copy()
    this_reg_y     = values_df.loc[values_df.index.intersection(this_reg_x.index)]

    this_clf_model = ml_helper.get_classifier_model()
    this_clf_param = ml_helper.get_classifier_model_cv_params()
    this_clf_x     = ml_helper.get_classifier_input_x().copy()
    this_clf_y     = values_df.loc[values_df.index.intersection(this_clf_x.index)]

    reg_models = generate_models_from_list(
        fields_to_model = continuous_values_cols,
        feature_df      = this_reg_x,
        value_df        = this_reg_y,
        scaler          = scaler,
        pca_obj         = pca_comb,
        model           = this_reg_model,
        cv_parameters   = this_reg_param,
        test_size       = 0.20,
        n_jobs          = 4,
        cv              = 3,
    )

    class_models = generate_models_from_list(
        fields_to_model = class_values_list,
        feature_df      = this_clf_x,
        value_df        = this_clf_y,
        scaler          = scaler,
        pca_obj         = pca_comb,
        model           = this_clf_model,
        cv_parameters   = this_clf_param,
        test_size       = 0.20,
        n_jobs          = 4,
        scoring         = 'precision_micro',
        cv              = 3,
        multiclass      = True,
        balance_sample  = 1.2
    )

    combined_models = {
        'reg_models': reg_models,
        'class_models': class_models,
        'propogate_cols': propogate_cols,
        'continuous_cols': continuous_values_cols,
        'propogate_cols': propogate_cols,
        'discreet_cols': class_values_ranges,
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

    scaled_joined_df = scale_combine_team_opposition(
        output_dfs,
        combined_models['reg_models'].get_scaler(),
        key_fields
    )

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

    team_pca_inp_df = team_rolling_df[key_fields+team_fields].merge(
        opposing_rolling_df[key_fields],
        on=key_fields
    ).sort_values(key_fields)
    opp_pca_inp_df = team_rolling_df[key_fields].merge(
        opposing_rolling_df[key_fields+opposing_fields],
        on=key_fields
    ).sort_values(key_fields)
    pca_data = combined_models['reg_models'].transform( team_pca_inp_df, opp_pca_inp_df )
    pca_data_cols = [str(i) for i in range(0,pca_data.shape[1])]
    pca_out_df = pd.DataFrame(pca_data,columns = pca_data_cols)

    output_df = values_out_df[key_fields+combined_models['propogate_cols']].copy()

    ml_helper = MLTrainingHelper(
        joined_out_df,
        scaled_joined_out_df,
        pca_out_df,
        input_arguments['reg_model_type'],
        input_arguments['clf_model_type'],
    )

    this_reg_x = ml_helper.get_regressor_input_x()
    this_clf_x = ml_helper.get_classifier_input_x()

    for col in combined_models['reg_models'].get_model_predicted_fields():
        output_df[col] = combined_models['reg_models'].predict(col,this_reg_x)

    for col in combined_models['class_models'].get_model_predicted_fields():
        output_df[col] = combined_models['class_models'].predict(col,this_clf_x)

    output_dict = {
        'output_result': output_df,
        'continuous_cols': combined_models['continuous_cols'],
        'propogate_cols':  combined_models['propogate_cols' ],
        'discreet_cols':   combined_models['discreet_cols'  ],
    }
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
