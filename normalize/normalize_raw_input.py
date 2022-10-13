#!/usr/bin/env python
import pandas as pd
import pickle as pkl
import argparse
import datetime
import os

__normalization_version__ = "0.1.0"

'''
Looks under input path for file starting with input string, that contains an input year
Combine all and output as dataframe
'''
def aggregate_read_data_files(inp_str,inp_path,inp_year_list):
    file_list=[]
    for fn in os.listdir(inp_path):
        if (fn.startswith(inp_str)):
            for year in inp_year_list:
                if (str(year) in fn):
                    file_list.append(pd.read_pickle(inp_path+fn))
    output_df = pd.concat(file_list)
    output_df = output_df.loc[
        ~output_df['posteam'].isnull()&
        ~output_df['defteam'].isnull()&
        (output_df['posteam']!='')&
        (output_df['defteam']!='')
     ]
    return output_df


'''
Rollup df on key fields, sum extra fields
'''
def rollup_df(inp_df,key_fields,extra_fields):
    return inp_df[
        key_fields + extra_fields
    ].groupby(key_fields).sum().reset_index()


def load_process_kick_data(raw_input_path,input_year_list):
    kicking_df = aggregate_read_data_files('kick_points',raw_input_path,input_year_list)
    kicking_df['team'] = kicking_df['posteam']

    kicking_df['extra_point_attempt'] = kicking_df['extra_point_attempt'].fillna(0).astype(int)
    kicking_df['extra_point_success'] = 0
    kicking_df.loc[kicking_df['extra_point_result']=='good','extra_point_success'] = 1

    kicking_df['field_goal_attempt'] = kicking_df['field_goal_attempt'].fillna(0).astype(int)
    kicking_df['field_goal_success'] = 0
    kicking_df.loc[kicking_df['field_goal_result']=='made','field_goal_success'] = 1

    kicking_df['kick_distance'] = kicking_df['kick_distance'].fillna(0).astype(int)

    kicking_df['kick_distance_0_39'] = 0
    kicking_df.loc[
        (kicking_df['field_goal_attempt']==1) &
        (kicking_df['kick_distance']<40),
        'kick_distance_0_39'
    ] = 1

    kicking_df['kick_distance_40_49'] = 0
    kicking_df.loc[
        (kicking_df['field_goal_attempt']==1) &
        (kicking_df['kick_distance_0_39']==0) &
        (kicking_df['kick_distance']<50),
        'kick_distance_40_49'
    ] = 1

    kicking_df['kick_distance_50'] = 0
    kicking_df.loc[
        (kicking_df['field_goal_attempt']==1) &
        (kicking_df['kick_distance_0_39']==0) &
        (kicking_df['kick_distance_40_49']==0) &
        (kicking_df['kick_distance']>=50),
        'kick_distance_50'
    ] = 1

    kicking_df['kick_distance_0_39_success' ] = 0
    kicking_df['kick_distance_40_49_success'] = 0
    kicking_df['kick_distance_50_success'   ] = 0

    kicking_df['kick_distance_0_39_success' ] = kicking_df['field_goal_success'] * kicking_df['kick_distance_0_39' ]
    kicking_df['kick_distance_40_49_success'] = kicking_df['field_goal_success'] * kicking_df['kick_distance_40_49']
    kicking_df['kick_distance_50_success'   ] = kicking_df['field_goal_success'] * kicking_df['kick_distance_50'   ]

    kicking_df['kick_distance_0_39_fail' ] = 0
    kicking_df['kick_distance_40_49_fail'] = 0
    kicking_df['kick_distance_50_fail'   ] = 0

    kicking_df['kick_distance_0_39_fail' ] = (1-kicking_df['field_goal_success']) * kicking_df['kick_distance_0_39' ]
    kicking_df['kick_distance_40_49_fail'] = (1-kicking_df['field_goal_success']) * kicking_df['kick_distance_40_49']
    kicking_df['kick_distance_50_fail'   ] = (1-kicking_df['field_goal_success']) * kicking_df['kick_distance_50'   ]

    return kicking_df


def load_process_yardage_data(raw_input_path,input_year_list):
    yardage_df = aggregate_read_data_files('play_yardage',raw_input_path,input_year_list)
    yardage_df['team'] = yardage_df['posteam']

    for yd in [
        'passing_yards', 'receiving_yards',
        'rushing_yards', 'return_yards',
        'lateral_receiving_yards', 'lateral_rushing_yards'
    ]:
        yardage_df[yd] = yardage_df[yd].fillna(0).astype(int)

    yardage_df['total_rushing_yards'  ] = yardage_df['rushing_yards'  ] + yardage_df['lateral_rushing_yards'  ]
    yardage_df['total_receiving_yards'] = yardage_df['receiving_yards'] + yardage_df['lateral_receiving_yards']

    yardage_df['passing_yards_40'] = 0
    yardage_df.loc[
        yardage_df['passing_yards']>=40,
        'passing_yards_40'
    ] = 1
    yardage_df['passing_yards_40'] = yardage_df['passing_yards_40'].astype(int)

    yardage_df['rushing_recieving_yards_40'] = 0
    yardage_df.loc[
        (yardage_df['rushing_yards'  ]>=40)|
        (yardage_df['receiving_yards']>=40),
        'rushing_recieving_yards_40'
    ] = 1
    yardage_df['rushing_recieving_yards_40'] = yardage_df['rushing_recieving_yards_40'].astype(int)

    return yardage_df


def load_process_touchdown_data(raw_input_path,input_year_list):
    td_df = aggregate_read_data_files('non_kick_points',raw_input_path,input_year_list).drop(
        columns=['defensive_extra_point_attempt', 'defensive_extra_point_conv',]
    )
    td_df['team'] = td_df['td_team']

    for td in [
        'touchdown', 'pass_touchdown',
        'rush_touchdown', 'return_touchdown', 'own_kickoff_recovery_td',
        'two_point_attempt',
        'defensive_two_point_attempt', 'defensive_two_point_conv'
    ]:
        td_df[td] = td_df[td].fillna(0).astype(int)

    td_df['two_point_conv_success'] = 0
    td_df.loc[td_df['two_point_conv_result']=='success','two_point_conv_success'] = 1

    return td_df


def load_process_turnover_data(raw_input_path,input_year_list):
    turnover_df = aggregate_read_data_files('turnovers',raw_input_path,input_year_list)
    turnover_df['team'] = turnover_df['posteam']

    for t in [
        'interception', 'fumble', 'fumble_forced', 'fumble_not_forced'
    ]:
        turnover_df[t] = turnover_df[t].fillna(0).astype(int)

    turnover_df['second_fumble'] = 0
    turnover_df.loc[~turnover_df['fumbled_2_team'].isnull(),'second_fumble'] = 1

    turnover_df['posteam_fumble_1'] = 0
    turnover_df['posteam_fumble_2'] = 0

    turnover_df['posteam_recovery_fumble_1'] = 0
    turnover_df['posteam_recovery_fumble_2'] = 0

    turnover_df.loc[turnover_df['fumbled_1_team']==turnover_df['posteam'],'posteam_fumble_1'] = 1
    turnover_df.loc[turnover_df['fumbled_2_team']==turnover_df['posteam'],'posteam_fumble_2'] = 1

    turnover_df.loc[turnover_df['fumble_recovery_1_team']==turnover_df['posteam'],'posteam_recovery_fumble_1'] = 1
    turnover_df.loc[turnover_df['fumble_recovery_2_team']==turnover_df['posteam'],'posteam_recovery_fumble_2'] = 1

    return turnover_df


def load_process_penalty_data(raw_input_path,input_year_list):
    penalty_df = aggregate_read_data_files('penalties',raw_input_path,input_year_list)
    penalty_df['team'] = penalty_df['penalty_team']

    for p in [
        'penalty', 'penalty_yards'
    ]:
        penalty_df[p] = penalty_df[p].fillna(0).astype(int)

    return penalty_df


def load_process_defense_data(raw_input_path,input_year_list):
    defense_df = aggregate_read_data_files('defensive',raw_input_path,input_year_list)
    defense_df['team'] = defense_df['defteam']

    for d in [
        'assist_tackle', 'qb_hit', 'solo_tackle', 'sack',
    ]:
        defense_df[d] = defense_df[d].fillna(0).astype(int)

    return defense_df


def load_all_raw_data(raw_input_path,input_year_list,key_fields):
    
    all_pos_def_teams = []

    print("Processing kicker data...")
    kick_df = load_process_kick_data(raw_input_path,input_year_list)
    all_pos_def_teams.append( kick_df[key_fields+['posteam','defteam']] )
    print('Done.')

    print("Processing yardage data...")
    yard_df = load_process_yardage_data(raw_input_path,input_year_list)
    all_pos_def_teams.append( yard_df[key_fields+['posteam','defteam']] )
    print("Done.")

    print("Processing endzone point data...")
    td_df = load_process_touchdown_data(raw_input_path,input_year_list)
    all_pos_def_teams.append( td_df[key_fields+['posteam','defteam']] )
    print("Done.")

    print("Processing turnover data...")
    turn_df = load_process_turnover_data(raw_input_path,input_year_list)
    all_pos_def_teams.append( turn_df[key_fields+['posteam','defteam']] )
    print("Done.")

    print("Processing penalty data...")
    plty_df = load_process_penalty_data(raw_input_path,input_year_list)
    all_pos_def_teams.append( plty_df[key_fields+['posteam','defteam']] )
    print("Done.")

    print("Processing defensive team data...")
    defn_df = load_process_defense_data(raw_input_path,input_year_list)
    all_pos_def_teams.append( defn_df[key_fields+['posteam','defteam']] )
    print("Done.")

    week_teams_playing = pd.concat(all_pos_def_teams).drop(columns=['team']).drop_duplicates().rename(
        columns={'posteam':'team','defteam':'opponent'}
    ).reset_index(drop=True)

    return {
        'matchup':week_teams_playing,
        'yardage':yard_df,
        'touchdowns':td_df,
        'kicks':kick_df,
        'turnovers':turn_df,
        'penalties':plty_df,
        'defensive':defn_df,
    }

def team_weekly_rollup( data_dict, key_fields ):

    week_teams_playing =  data_dict['matchup']
    yard_df            =  data_dict['yardage']
    td_df              =  data_dict['touchdowns']
    kick_df            =  data_dict['kicks']
    turn_df            =  data_dict['turnovers']
    plty_df            =  data_dict['penalties']
    defn_df            =  data_dict['defensive']

    kick_df_rollup = rollup_df(
        kick_df,
        key_fields,
        [
            'extra_point_attempt', 'extra_point_success',
            'field_goal_attempt', 'field_goal_success',
            'kick_distance_0_39_success', 'kick_distance_40_49_success', 'kick_distance_50_success',
            'kick_distance_0_39_fail', 'kick_distance_40_49_fail', 'kick_distance_50_fail'
        ]
    )

    yard_df['rushing_yards'] = yard_df['total_rushing_yards']
    yard_df['receiving_yards'] = yard_df['total_receiving_yards']
    yard_df_rollup = rollup_df(
        yard_df,
        key_fields,
        [
            'rushing_yards', 'passing_yards', 'receiving_yards', 'return_yards',
            'passing_yards_40', 'rushing_recieving_yards_40'
        ]
    )

    high_yards_plays = yard_df[key_fields+['game_id','play_id','rushing_recieving_yards_40']]
    td_yard_df = td_df[
        key_fields + [
            'game_id', 'play_id',
            'touchdown', 'pass_touchdown',
            'rush_touchdown', 'return_touchdown', 'own_kickoff_recovery_td',
            'two_point_attempt', 'two_point_conv_success',
            'defensive_two_point_attempt', 'defensive_two_point_conv'
        ]
    ].merge(
        high_yards_plays,
        on=key_fields+['game_id','play_id'],
        how='left'
    ).fillna(0)
    td_yard_df['td_yards_40'] = td_yard_df['rushing_recieving_yards_40'].astype(int)
    td_rollup = rollup_df(
        td_yard_df,
        key_fields,
        [
            'touchdown', 'pass_touchdown',
            'rush_touchdown', 'return_touchdown',
            'two_point_attempt', 'two_point_conv_success','td_yards_40'
        ]
    )

    interception_df = turn_df[key_fields+[
        'interception'
    ]]
    interception_df_rollup = rollup_df(
        interception_df,
        key_fields,
        [
            'interception',
        ]
    )

    fumb1_df = turn_df[key_fields+[
        'fumble','fumble_forced','fumble_not_forced',
        'fumbled_1_team',
    ]].copy()
    fumb1_df['team'] = fumb1_df['fumbled_1_team']

    fumb2_df = turn_df.loc[turn_df['second_fumble']==1,key_fields+[
        'fumbled_2_team',
    ]].copy()
    fumb2_df['team'] = fumb2_df['fumbled_2_team']
    fumb2_df['fumble'] = 1
    fumb2_df['fumble_forced'] = 0
    fumb2_df['fumble_not_forced'] = 1

    fumb_df = pd.concat(
        [
            fumb1_df[key_fields+['fumble','fumble_forced','fumble_not_forced']],
            fumb2_df[key_fields+['fumble','fumble_forced','fumble_not_forced']],
        ],ignore_index=True
    ).reset_index()
    fumb_df_rollup = rollup_df(
        fumb_df,
        key_fields,
        [
            'fumble','fumble_forced','fumble_not_forced',
        ]
    )

    penalty_df_rollup = rollup_df(
        plty_df,
        key_fields,
        [
            'penalty', 'penalty_yards',
        ]
    )

    defense_df_rollup = rollup_df(
        defn_df,
        key_fields,
        [
            'assist_tackle', 'qb_hit', 'solo_tackle', 'sack',
        ]
    )

    full_weekly_team_join = week_teams_playing.sort_values(['season','week','team']).reset_index(drop=True)
    for rollup in [
        yard_df_rollup,
        td_rollup,
        kick_df_rollup,
        interception_df_rollup,
        fumb_df_rollup,
        penalty_df_rollup,
        defense_df_rollup,
    ]:
        full_weekly_team_join = full_weekly_team_join.merge(
            rollup,
            on=key_fields,
            how='left'
        )
    full_weekly_team_join = full_weekly_team_join.fillna(0)

    full_weekly_team_join['points'] = (
        6*full_weekly_team_join['touchdown']+
        2*full_weekly_team_join['two_point_conv_success']+
        1*full_weekly_team_join['extra_point_success']+
        3*full_weekly_team_join['field_goal_success']
    )

    opposing_df = full_weekly_team_join[['season','week','team','opponent','points']].rename(
        columns={'team':'opponent','opponent':'team','points':'opponent_points'}
    )
    opposing_df.head()

    full_weekly_team_join = full_weekly_team_join.merge(
        opposing_df,
        on=['season','week','team','opponent'],
        how='left',
    )
    full_weekly_team_join['win'] = (full_weekly_team_join['points']>full_weekly_team_join['opponent_points']).astype(int)

    return full_weekly_team_join


def get_normalized_data_path(version=__normalization_version__):
    return os.getcwd()+'/data/normalized/'+version+'/'


def write_normalized_team_data(input_df,write_dir):
    os.makedirs(write_dir,exist_ok=True)
    print("Outputting data to "+write_dir)

    for season in input_df['season'].unique():
        this_df = input_df.loc[input_df['season']==season].reset_index(drop=True)
        fn = "weekly_team_data_season_"+str(season)+".pkl"
        full_output_fn_path = write_dir+fn
        print("Writing "+fn+" to file")
        print("Season shape: ",this_df.shape)
        this_df.to_pickle(full_output_fn_path)
        print("Wrote "+full_output_fn_path)

# Only start and end year, can read from default max range
def read_args():
    
    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

    parser.add_argument('--start_year', type=int, nargs='?', default=1999,
        help='The starting year for input data (Earliest available data is from 1999)'
    )
    parser.add_argument('--end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for input data (Latest is current year)'
    )

    args = parser.parse_args()
    print(args)
    return vars(args)

def main():
    input_arguments = read_args()

    #TODO: change project path to be better handled
    project_path = os.getcwd()+"/"#'/home/sean/Documents/Fantasy_Football/'

    #TODO: properly implement raw versioning here
    raw_input_path = project_path+'data/raw/0.1.0/'

    assert input_arguments['start_year']>=1999 # Earliest year is 1999
    assert input_arguments['end_year']<=datetime.date.today().year # Latest year cannot exceed current
    assert input_arguments['start_year']<=input_arguments['end_year'] # Start year cannot be greater than end year

    input_year_list = [input_arguments['start_year']]
    for i in range(input_arguments['start_year']+1,input_arguments['end_year']+1):
        input_year_list.append(i)

    # Fields for joining weekly team data on
    key_fields = ['season','week','team']
        
    inputs = load_all_raw_data(raw_input_path,input_year_list,key_fields)
    
    team_rollup = team_weekly_rollup( inputs, key_fields )
    write_normalized_team_data(team_rollup,get_normalized_data_path())
    
if __name__ == "__main__":
    main()
