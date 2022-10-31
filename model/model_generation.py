import sys
#TODO: modify this
sys.path.append('/home/sean/Documents/import_test/util')

#from util.utilities import PCACols,ZScaler
from sklearn.decomposition import PCA

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import pickle as pkl
import numpy as np
import pandas as pd
import os
import argparse
import datetime

__model_version__='0.1.0'

#TODO: should be in utils, copy this version
class ZScaler:
    def __init__(self,inp_df,columns=None):
        self.scale_dict = {}
        assert isinstance(inp_df,pd.DataFrame) # Input must be a dataframe
        if (columns is None):
            self.add(inp_df)
        else:
            if (not isinstance(columns,list)):
                columns = [columns]
            self.add(inp_df[columns])

    def __repr__(self):
        out_str = ""
        for key in self.scale_dict:
            out_str += "\tField: {:50s}\tMean: {:10.6f}\tStd: {:10.6f}\n".format(
                key,
                self.scale_dict[key]['mean'],
                self.scale_dict[key]['std'],
            )
        return out_str

    def __str__(self):
        return "Member of ZScaler\n"+self.__repr__()

    def add(self,inp_df):
        for col in inp_df.columns.values:
            self.scale_dict[col] = {}
            self.scale_dict[col]['mean'] = inp_df[col].mean()
            self.scale_dict[col]['std'] = inp_df[col].std()

    def remove(self,col):
        del self.scale_dict[col]

    def get_dict(self):
        return self.scale_dict

    def get(self,col,kind=None):
        if (kind is None):
            return self.scale_dict[col]
        elif((kind=='mean') or (kind=='std')):
            return self.scale_dict
        else:
            raise ValueError("kind must be 'mean' or 'std'")

    def scale_cols(self,inp_df,cols):
        out_df = inp_df.copy()
        for col in cols:
            assert col in self.scale_dict # Can only scale columns in the scaler!
            out_df[col]=(out_df[col]-self.scale_dict[col]['mean'])/self.scale_dict[col]['std']
            out_df.rename(columns={col:col+'_z'},inplace=True)
        return out_df

#TODO: move w/ ^
class PCACols:
    def __init__(self,inp_df,columns,n_components):
        self.columns = columns
        self.n_components = n_components
        self.model_pca = PCA(n_components=n_components)
        self.model_pca.fit(inp_df[self.columns])

    def __repr__(self):
        out_str = ""
        out_str+= "\tN components = "+str(self.n_components)+"\n"
        out_str+= "\tColumns = [\n"
        for col in self.columns:
            out_str+="\t\t"+str(col)+",\n"
        out_str+= "\t]\n"
        out_str+= "\tCumulative Explained Variance = [\n"
        for var in self.model_pca.explained_variance_ratio_.cumsum():
            out_str+="\t\t"+str(var)+",\n"
        out_str+= "\t]\n"
        return out_str

    def __str__(self):
        return "Member of PLCACols\n"+self.__repr__()

    def PCA(self):
        return self.model_pca

    def transform(self,inp_df):
        return self.model_pca.transform(inp_df[self.columns])

class ModelWrapper:
    def __init__(
        self,
        input_x_df,
        input_y_df,
        key_fields=['season','week','team','opponent']
    ):
        self.x_df = input_x_df.dropna()
        self.y_df = input_y_df.loc[self.x_df.index].dropna()
        assert self.x_df.shape[0]==self.y_df.shape[0] # X and Y must have same dimension
        self.key_fields = key_fields
        self.model_dict = {}
        self.col_dict = {}
        self.cv_dict = {}

    def __get_values__(self,input_df,cols):
        return input_df.drop(columns=self.key_fields)[cols].values

    def get_model_dict(self):
        return self.model_dict

    def get_model_predicted_fields(self):
        return self.col_dict.keys()

    def get_model_predictor_fields(self):
        return self.col_dict

    def get_cv_dict(self):
        return self.cv_dict

    def model(self,name):
        assert name in self.model_dict
        return self.model_dict[name]

    def train_model(
        self,
        names,
        model,
        parameters=None,
        use_cols=None,
        test_size=0.20,
        n_jobs=1,
        scoring=None,
        cv=3,
    ):
        if isinstance(names,str):
            names=[names]
        assert isinstance(names,list)
        for name in names:
            assert name in self.y_df.columns.values

        for name in names:
            if (use_cols is None):
                use_cols = self.x_df.drop(columns=self.key_fields).columns.values
            self.col_dict[name] = use_cols
            features = self.__get_values__(self.x_df,self.col_dict[name])
            values   = self.__get_values__(self.y_df,name)

            x_shuf, y_shuf = shuffle( features, values )
            x_train, x_test, y_train, y_test = train_test_split( x_shuf, y_shuf, test_size=test_size )

            if (parameters is None):
                self.model_dict[name] = model.fit(x_train,y_train)
            else:
                gscv = GridSearchCV( model, parameters, n_jobs=n_jobs, scoring=scoring,cv=cv )
                gscv.fit(x_train, y_train)
                self.cv_dict[name] = gscv
                self.model_dict[name] = gscv.best_estimator_.fit(x_train,y_train)

            print("Fit model for "+name+", test data score=",str(self.model_dict[name].score(x_test,y_test)))

    def predict(self,name,inp_df):
        assert isinstance(name,str)
        assert name in self.model_dict

        features = self.__get_values__(inp_df,self.col_dict[name])
        return self.model_dict[name].predict(features)

    def predict_proba(self,name,inp_df):
        assert isinstance(name,str)
        assert name in self.model_dict

        features = self.__get_values__(inp_df,self.col_dict[name])
        return self.model_dict[name].predict_proba(features)

def aggregate_read_data_files(inp_str,inp_path,inp_year_list):
    file_list=[]
    for fn in os.listdir(inp_path):
        if (fn.startswith(inp_str)):
            for year in inp_year_list:
                if (str(year) in fn):
                    file_list.append(pd.read_pickle(inp_path+fn))
    output_df = pd.concat(file_list).sort_values(['season','week','team']).drop_duplicates().reset_index(drop=True)
    return output_df


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
    weekly_df['close_field_goal_attempts'] = (
        weekly_df['kick_distance_0_39_success'] + weekly_df['kick_distance_0_39_fail']
    )
    weekly_df['close_field_goal_success_rate'] = weekly_df['kick_distance_0_39_success'] / (
        weekly_df['close_field_goal_attempts'] + 1e-7
    )

    weekly_df['kick_distance_40_success'] = weekly_df['kick_distance_40_49_success'] + weekly_df['kick_distance_50_success']
    weekly_df['kick_distance_40_fail'] = weekly_df['kick_distance_40_49_fail'] + weekly_df['kick_distance_50_fail']

    weekly_df['far_field_goal_attempts'] = (
        weekly_df['kick_distance_40_success'] + weekly_df['kick_distance_40_fail']
    )
    weekly_df['far_field_goal_success_rate'] = weekly_df['kick_distance_40_success'] / (
        weekly_df['far_field_goal_attempts'] + 1e-7
    )

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

def filter_df_year(inp_df,start_year,end_year):
    return inp_df.loc[
        (inp_df['season']>=start_year) &
        (inp_df['season']<=  end_year)
    ]

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

    rolling_scaled_weekly_fields = []
    for field in rolling_scaled_weekly_df.columns.values:
        if (field not in key_fields):
            rolling_scaled_weekly_fields.append(field)

    opposing_scaled_weekly_fields = []
    for field in opposing_scaled_weekly_df.columns.values:
        if (field not in key_fields):
            opposing_scaled_weekly_fields.append(field)

    joined_weekly_df = rolling_scaled_weekly_df.merge(
        opposing_scaled_weekly_df,
        on=['season','week','team','opponent']
    )

    return {
        'joined_weekly_df': joined_weekly_df,
        'team_fields': rolling_scaled_weekly_fields,
        'opposing_fields': opposing_scaled_weekly_fields,
    }

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
            cv=cv
        )
    return my_models


#TODO:change
def get_model_path(version=__model_version__):
    return os.getcwd()+'/data/model/'+version+'/'

def get_prediction_path(version=__model_version__):
    return os.getcwd()+'/data/predicted_values/'+version+'/'


#def write_normalized_team_data(input_df,write_dir):
#    os.makedirs(write_dir,exist_ok=True)
#    print("Outputting data to "+write_dir)
#
#    for season in input_df['season'].unique():
#        this_df = input_df.loc[input_df['season']==season].reset_index(drop=True)
#        fn = "weekly_team_data_season_"+str(season)+".pkl"
#        full_output_fn_path = write_dir+fn
#        print("Writing "+fn+" to file")
#        print("Season shape: ",this_df.shape)
#        this_df.to_pickle(full_output_fn_path)
#        print("Wrote "+full_output_fn_path)


# Only start and end year, can read from default max range
def read_args():

    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

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

    parser.add_argument('--output_scaler_file_name', type=str, nargs='?',
        help='Optional scaler file to save, otherwise uses date'
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

    parser.add_argument('--output_data_file_name', type=str, nargs='?',
        help='Optional save data with prediction'
    )

    args = parser.parse_args()
    print(args)

    input_arguments = vars(args)

    assert input_arguments['norm_start_year']>=1999 # Earliest year is 1999
    assert input_arguments['norm_end_year']<=datetime.date.today().year # Latest year cannot exceed current
    assert input_arguments['norm_start_year']<=input_arguments['norm_end_year'] # Start year cannot be greater than end year
    assert input_arguments['process_start_year']>=1999 # Earliest year is 1999
    assert input_arguments['process_end_year']<=datetime.date.today().year # Latest year cannot exceed current
    assert input_arguments['process_start_year']<=input_arguments['process_end_year'] # Start year cannot be greater than end year
    assert input_arguments['predict_start_year']>=1999 # Earliest year is 1999
    assert input_arguments['predict_end_year']<=datetime.date.today().year # Latest year cannot exceed current
    assert input_arguments['predict_start_year']<=input_arguments['predict_end_year'] # Start year cannot be gr    assert input_arguments['n_rolling'] > 0 # Lookback months must be greater than 0
    assert input_arguments['n_components_team'] > 0 # Must be greater than 0
    assert input_arguments['n_components_opp'] > 0 # Must be greater than 0

    if (input_arguments['input_scaler_file_name'] is not None):
        fn = get_model_path()+input_arguments['input_scaler_file_name']
        if ( not os.path.exists(fn) ):
            raise IOError("File does not exist: "+fn)

    if (input_arguments['input_models_file_name'] is not None):
        fn = get_model_path()+input_arguments['input_models_file_name']
        if ( not os.path.exists(fn) ):
            raise IOError("File does not exist: "+fn)

    if (input_arguments['predict_values']):
        assert input_arguments['output_data_file_name'] is not None # If outputting data, must have file name

    return input_arguments

def main():

    input_arguments = read_args()

    #TODO: change project path to be better handled
    project_path = os.getcwd()+"/"#'/home/sean/Documents/Fantasy_Football/'

    #TODO: properly implement raw versioning here
    norm_input_path = project_path+'data/normalized/0.1.0/'

    input_year_list = [input_arguments['norm_start_year']]
    for i in range(input_arguments['norm_start_year']+1,input_arguments['norm_end_year']+1):
        input_year_list.append(i)

    norm_weekly_df = aggregate_read_data_files("weekly_team_data_season",norm_input_path,input_year_list)


    key_fields = ['season','week','team','opponent']

    output_dfs = generate_weekly_team_features_values( norm_weekly_df, key_fields, input_arguments['n_rolling'] )

    team_fields         = output_dfs['team_fields']
    opposing_fields     = output_dfs['opposing_fields']

    values_df           = filter_df_year( output_dfs['value_df']   , input_arguments['process_start_year'], input_arguments['process_end_year'] )
    team_rolling_df     = filter_df_year( output_dfs['team_df']    , input_arguments['process_start_year'], input_arguments['process_end_year'] )
    opposing_rolling_df = filter_df_year( output_dfs['opposing_df'], input_arguments['process_start_year'], input_arguments['process_end_year'] )


    joined_df = team_rolling_df[key_fields+team_fields].merge(
        opposing_rolling_df[key_fields+opposing_fields],
        on=key_fields
    )
    features_df = joined_df.drop(columns=['season','week','team','opponent']).dropna()

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

    scaled_dict = scale_combine_team_opposition( output_dfs, scaler, key_fields )

    scaled_joined_df = scaled_dict['joined_weekly_df']
    scaled_features_df = scaled_joined_df.drop(columns=['season','week','team','opponent']).dropna()

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
        'rushing_yards', 'receiving_yards',
    ]

    propogate_cols = [
        'close_field_goal_success_rate',
        'far_field_goal_success_rate',
        'fumble_recovery_rate',
    ]

    class_values_ranges = {
        'complete_pass':[0,10,15,20,25,30,35],
        'touchdown':[0,1,2,3,4,5],
        'pass_touchdown':[0,1,2,3,4],
        'rush_touchdown':[0,1,2,3,4],
        'td_yards_40':[0,1,2],
        'close_field_goal_attempts':[0,1,2,3,4],
        'far_field_goal_attempts':[0,1,2,3],
        'sack':[0,2,4,6,8],
        'fumble':[0,2,4,6],
        'offensive_interception':[0,2,4],
        'defensive_points_allowed':[0,1,7,14,21,28,35],
        'defensive_fumble_forced':[0,2,4],
        'defensive_interception':[0,2,4],
    }

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

    if (input_arguments['input_models_file_name'] is None):
        reg_models = generate_models_from_list(
            fields_to_model=continuous_values_cols,
            feature_df = joined_df,
            value_df = values_df,
            model = LinearRegression(),
            test_size=0.20,
            n_jobs=8,
            cv=3,
        )
        print(reg_models.model(continuous_values_cols[0]))

        class_models = generate_models_from_list(
            fields_to_model=class_values_list,
            feature_df = scaled_joined_df,
            value_df = values_df,
            model = LogisticRegression(),
            test_size=0.20,
            n_jobs=8,
            cv=3,
        )

        combined_models = {
            'reg_models': reg_models,
            'class_models': class_models,
        }

        if (input_arguments['output_models_file_name'] is not None):
            output_name = get_model_path() + input_arguments['output_models_file_name']
        else:
            output_name = get_model_path() + \
                "models_"+\
                str(input_arguments["process_start_year"])+\
                "_"+\
                str(input_arguments["process_end_year"])+\
                ".pkl"
        os.makedirs(get_model_path(),exist_ok=True)
        with open(output_name, 'wb') as f:
            pkl.dump(combined_models,f)
        print("Wrote file "+output_name)

    else:
        input_name = get_model_path()+input_arguments['input_models_file_name']
        with open(input_name,'rb') as f:
            combined_models = pkl.load(f)
        print("Using file "+input_name)

    if (input_arguments['predict_values']):

        scaled_joined_out_df = filter_df_year(
            scaled_joined_df,
            input_arguments['predict_start_year'],
            input_arguments['predict_end_year']
        ).dropna()

        joined_out_df = filter_df_year(
            joined_df,
            input_arguments['predict_start_year'],
            input_arguments['predict_end_year']
        ).dropna().iloc[scaled_joined_out_df.index]

        values_out_df = filter_df_year(
            values_df,
            input_arguments['predict_start_year'],
            input_arguments['predict_end_year']
        ).dropna().iloc[scaled_joined_out_df.index]

        output_dict = {
            'propogate_df':values_out_df[key_fields+propogate_cols],
        }

        for col in combined_models['reg_models'].get_model_predicted_fields():
            output_dict[col] = combined_models['reg_models'].predict(col,joined_out_df)

        for col in combined_models['class_models'].get_model_predicted_fields():
            output_dict[col] = combined_models['class_models'].predict_proba(col,scaled_joined_out_df)

        os.makedirs(get_model_path(),exist_ok=True)
        output_name = get_model_path() + input_arguments['output_data_file_name']
        with open(output_name, 'wb') as f:
            pkl.dump(output_dict,f)
        print("Wrote "+output_name)

if __name__ == "__main__":
    main()
