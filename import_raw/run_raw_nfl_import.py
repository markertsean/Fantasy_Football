#!/usr/bin/env python
import nfl_data_py as nfl
import pandas as pd
import datetime
import argparse
import os
import pickle as pkl
import sys

sys.path.append('/home/sean/Documents/Fantasy_Football/')

from util import utilities
from util import argument_validation

__import_version__ = '0.1.0'

# Only start and end year, can read from default max range
def __read_args__():

    parser = argparse.ArgumentParser(description='Read and save data from nfl_data_py requests using input years')

    parser.add_argument('--ingest_start_year', type=int, nargs='?', default=1999,
        help='The starting year for ingestion data (Earliest available data is from 1999)'
    )
    parser.add_argument('--ingest_end_year', type=int, nargs='?',default=datetime.date.today().year,
        help='The ending year for ingestion data (Latest is current year)'
    )
    parser.add_argument('--input_version', type=str, nargs='?',
        default=__import_version__,
        help='The version to use for import'
    )

    args = parser.parse_args()
    inp_args = vars(args)

    inp_args['ingest'] = True

    argument_validation.run_argument_validation(inp_args)

    return inp_args

def __write_raw_data__(input_dict,write_dir):
    os.makedirs(write_dir,exist_ok=True)
    print("Outputting data to "+write_dir)

    for key in input_dict:
        this_df = input_dict[key]
        for year in this_df['season'].unique():
            year_df = this_df.loc[this_df['season']==year]
            full_output_fn_path = write_dir+key+"_"+str(year)+".pkl"
            print("Writing "+key+" to file")
            print(key+" shape: ",year_df.shape)
            year_df.to_pickle(full_output_fn_path)
            print("Wrote "+full_output_fn_path)

def get_raw_data_path(version=__import_version__):
    return os.getcwd()+'/data/raw/'+version+'/'

'''
Runs the import from nfl_data_py for the given year range
Generates df of team data aggregated from player weekly stats,
as well as play by play information for stats where yardage matters

Returns everything as a dictionary with names/dataframes
'''
def run_nfl_import( input_year_list ):

    print("Considering downloads for years:",input_year_list)
    all_weekly_game_data = nfl.import_weekly_data(
        input_year_list,
        downcast=True
    )
    print("Done.")

    print("Aggregating weekly data...")
    team_weekly_data = all_weekly_game_data[
        [
            'season_type', 'season','recent_team','week',
            'passing_tds', 'rushing_tds', 'special_teams_tds',
            'completions', 'attempts', 'passing_yards',
            'carries', 'rushing_yards',
            'passing_2pt_conversions', 'rushing_2pt_conversions',
            'sacks',
            'sack_fumbles', 'rushing_fumbles', 'receiving_fumbles',
            'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost',
            'interceptions',
        ]
    ].groupby(['season','week','recent_team']).sum().reset_index()
    print("Done.")

    # Need to generate weekly stats for teams
    #player_data = all_weekly_game_data[
    #    [
    #        'player_id', 'player_name', 'player_display_name',
    #        'position', 'position_group',
    #        'season_type', 'season','recent_team','week',
    #
    #        'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_air_yards', 'passing_yards_after_catch',
    #
    #        'carries', 'rushing_yards', 'rushing_tds',
    #
    #        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
    #        'receiving_air_yards', 'receiving_yards_after_catch',
    #
    #        'passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions',
    #
    #        'sacks', 'sack_yards',
    #        'sack_fumbles', 'sack_fumbles_lost',
    #        'rushing_fumbles', 'rushing_fumbles_lost',
    #        'receiving_fumbles', 'receiving_fumbles_lost',
    #        'interceptions',
    #
    #        'special_teams_tds',
    #        'fantasy_points', 'fantasy_points_ppr'
    #    ]
    #]


    play_by_play_cols = [
        'play_id', 'game_id', 'old_game_id',
        'home_team', 'away_team',
        'season_type', 'week',
        'posteam', 'posteam_type', 'defteam',
        'side_of_field', 'yardline_100',
        'goal_to_go', 'time', 'yrdln', 'ydstogo',
        'desc', 'play_type', 'yards_gained',

        'season', 'weather', 'temp', 'wind',

        'penalty','penalty_team', 'penalty_yards',

        'td_team', 'touchdown', 'pass_touchdown', 'rush_touchdown', 'return_touchdown',
        'own_kickoff_recovery_td','two_point_conv_result','tackle_with_assist',
        'defensive_two_point_attempt', 'defensive_two_point_conv',
        'defensive_extra_point_attempt', 'defensive_extra_point_conv',

        'pass', 'rush', 'special',
        'rush_attempt', 'pass_attempt', 'two_point_attempt',
        'field_goal_attempt', 'extra_point_attempt', 'kickoff_attempt', 'punt_attempt',

        'pass_length', 'air_yards', 'yards_after_catch',
        'passing_yards', 'receiving_yards', 'lateral_receiving_yards',
        'rushing_yards', 'lateral_rushing_yards',
        'complete_pass', 'incomplete_pass', 'interception',
        'lateral_reception',

        'extra_point_result', 'field_goal_result', 'kick_distance',
        'punt_blocked','return_yards',
        'lateral_return',

        'lateral_rush',

        'special_teams_play', 'st_play_type',

        'fumble', 'fumble_forced', 'fumble_not_forced',
        'lateral_recovery',

        'assist_tackle','qb_hit', 'solo_tackle', 'safety', 'sack',

        'td_player_id',
        'passer_player_id','receiver_player_id',
        'lateral_receiver_player_id',
        'rusher_player_id','lateral_rusher_player_id',
        'lateral_sack_player_id',
        'interception_player_id','lateral_interception_player_id',
        'punt_returner_player_id','lateral_punt_returner_player_id',
        'kickoff_returner_player_id','lateral_kickoff_returner_player_id',
        'punter_player_id','kicker_player_id',
        'own_kickoff_recovery_player_id',
        'blocked_player_id',
        'tackle_for_loss_1_player_id','tackle_for_loss_2_player_id',
        'qb_hit_1_player_id','qb_hit_2_player_id',
        'forced_fumble_player_1_team','forced_fumble_player_1_player_id',
        'forced_fumble_player_2_team','forced_fumble_player_2_player_id',
        'solo_tackle_1_player_id','solo_tackle_2_player_id',
        'assist_tackle_1_player_id','assist_tackle_2_player_id',
        'assist_tackle_3_player_id','assist_tackle_4_player_id',
        'tackle_with_assist_1_player_id','tackle_with_assist_2_player_id',
        'pass_defense_1_player_id','pass_defense_2_player_id',
        'fumbled_1_team', 'fumbled_1_player_id',
        'fumbled_2_team', 'fumbled_2_player_id',
        'fumble_recovery_1_team', 'fumble_recovery_1_player_id',
        'fumble_recovery_2_team', 'fumble_recovery_2_player_id',
        'sack_player_id',
        'half_sack_1_player_id','half_sack_2_player_id',
        'penalty_player_id',
        'safety_player_id',
        'passer_id', 'rusher_id', 'receiver_id',
    ]

    print("Downloading play by play data, this may take some time...")
    pbp_data_sample = nfl.import_pbp_data(
        input_year_list,
        columns=play_by_play_cols,
        downcast=True,
        cache=False,
        alt_path=None
    )
    print("Done.")


    '''
       Kickers:
           50+ yard FG 5
           40-49 FG 4
           39- FG 3
           Rush/pass/recep/2-pt conv 2
           Extra point made 1
           Penalty missed 0-39 -2
           Penalty missed 40-49 -1
    '''
    pbp_data_sample['kick_points'] = 0
    for col in [
        'field_goal_attempt','extra_point_attempt'
    ]:
        pbp_data_sample.loc[pbp_data_sample[col]!=0.0,'kick_points'] = 1

    fg_ep_data = pbp_data_sample.loc[
        pbp_data_sample['kick_points']==1,
        [
            'season', 'week','posteam','defteam',
            'play_id', 'game_id', 'old_game_id',
            'field_goal_attempt','extra_point_attempt',
            'extra_point_result','field_goal_result','kick_distance',
            'kicker_player_id'
        ]
    ]


    '''
        Offensive-
           passying yards 1/25yds
           passing tds 4
           rushing yards 1/10yds
           rushing tds 6
           Receptions 0.5 half, 1 full ppr
           Receiving yards 1/10yds
           receiving tds 6
           2-point conversions 2
           Fumble recovery for tds 6
           Bonus - rushing or receiving tds 40+ yds 2
           Bonus passing td of 40+ yds 2
           Penalty intercepted -1 or -2
           Fumble lost -2
    '''
    pbp_data_sample['any_yards'] = 0
    for col in [
        'passing_yards', 'receiving_yards', 'lateral_receiving_yards',
        'rushing_yards', 'lateral_rushing_yards','return_yards'
    ]:
        pbp_data_sample.loc[pbp_data_sample[col]!=0.0,'any_yards'] = 1

    play_yard_data = pbp_data_sample.loc[
        pbp_data_sample['any_yards']==1,
        [
            'season', 'week','posteam','defteam',
            'play_id', 'game_id', 'old_game_id',
            'passing_yards', 'receiving_yards', 'lateral_receiving_yards',
            'rushing_yards', 'lateral_rushing_yards','return_yards',
            'rush_attempt', 'pass_attempt', 'complete_pass',

            'passer_player_id','receiver_player_id','lateral_receiver_player_id',
            'rusher_player_id','lateral_rusher_player_id',
            'punt_returner_player_id','lateral_punt_returner_player_id',
            'kickoff_returner_player_id','lateral_kickoff_returner_player_id',
            'own_kickoff_recovery_player_id',
            'passer_id', 'rusher_id', 'receiver_id',
        ]
    ]



    pbp_data_sample['non_kick_point'] = 0
    for col in ['touchdown','two_point_attempt','defensive_two_point_attempt','defensive_extra_point_attempt']:
        pbp_data_sample.loc[pbp_data_sample[col]>0.0,'non_kick_point'] = 1

    td_data = pbp_data_sample.loc[
        pbp_data_sample['non_kick_point']==1,
        [
            'season', 'week','posteam','defteam',
            'play_id', 'game_id', 'old_game_id',
            'td_team', 'touchdown', 'pass_touchdown', 'rush_touchdown', 'return_touchdown',
            'own_kickoff_recovery_td','two_point_attempt','two_point_conv_result',
            'defensive_two_point_attempt', 'defensive_two_point_conv',
            'defensive_extra_point_attempt', 'defensive_extra_point_conv',

            'td_player_id',
            'passer_player_id','passer_id',
            'receiver_player_id','receiver_id','lateral_receiver_player_id',
            'rusher_player_id','rusher_id','lateral_rusher_player_id',
            'punt_returner_player_id','lateral_punt_returner_player_id',
            'kickoff_returner_player_id','lateral_kickoff_returner_player_id',
            'own_kickoff_recovery_player_id',
        ]
    ]



    pbp_data_sample['turnover'] = 0
    for col in ['interception', 'fumble', 'fumble_forced', 'fumble_not_forced']:
        pbp_data_sample.loc[pbp_data_sample[col]>0.0,'turnover'] = 1

    turnover_data = pbp_data_sample.loc[
        pbp_data_sample['turnover']==1,
        [
            'season', 'week','posteam','defteam',
            'play_id', 'game_id', 'old_game_id',
            'interception', 'fumble', 'fumble_forced', 'fumble_not_forced',
            'passer_id', 'rusher_id', 'receiver_id',
            'passer_player_id','receiver_player_id',
            'rusher_player_id','lateral_rusher_player_id',
            'interception_player_id','lateral_interception_player_id',
            'punt_returner_player_id','lateral_punt_returner_player_id',
            'kickoff_returner_player_id','lateral_kickoff_returner_player_id',
            'own_kickoff_recovery_player_id',
            'forced_fumble_player_1_team','forced_fumble_player_1_player_id',
            'forced_fumble_player_2_team','forced_fumble_player_2_player_id',
            'fumbled_1_team', 'fumbled_1_player_id',
            'fumbled_2_team', 'fumbled_2_player_id',
            'fumble_recovery_1_team', 'fumble_recovery_1_player_id',
            'fumble_recovery_2_team', 'fumble_recovery_2_player_id',
        ]
    ]


    penalty_data = pbp_data_sample.loc[
        pbp_data_sample['penalty']>0,
        [
            'season', 'week','posteam','defteam',
            'play_id', 'game_id', 'old_game_id',
            'penalty','penalty_team', 'penalty_yards',
            'penalty_player_id',
        ]
    ]


    pbp_data_sample['def_play'] = 0
    for col in ['punt_blocked','assist_tackle','qb_hit','solo_tackle','sack']:
        pbp_data_sample.loc[pbp_data_sample[col]>0.0,'def_play'] = 1

    defense_data = pbp_data_sample.loc[
        pbp_data_sample['def_play']==1,
        [
            'season', 'week','posteam','defteam',
            'play_id', 'game_id', 'old_game_id',

            'assist_tackle', 'qb_hit', 'solo_tackle', 'sack',

            'passer_player_id', 'passer_id',
            'receiver_player_id', 'receiver_id', 'lateral_receiver_player_id',
            'rusher_player_id', 'rusher_id', 'lateral_rusher_player_id',

            'lateral_sack_player_id',
            'interception_player_id','lateral_interception_player_id',
            'punt_returner_player_id','lateral_punt_returner_player_id',
            'kickoff_returner_player_id','lateral_kickoff_returner_player_id',
            'tackle_for_loss_1_player_id','tackle_for_loss_2_player_id',
            'qb_hit_1_player_id','qb_hit_2_player_id',
            'solo_tackle_1_player_id','solo_tackle_2_player_id',
            'assist_tackle_1_player_id','assist_tackle_2_player_id',
            'assist_tackle_3_player_id','assist_tackle_4_player_id',
            'tackle_with_assist_1_player_id','tackle_with_assist_2_player_id',
            'pass_defense_1_player_id','pass_defense_2_player_id',
            'sack_player_id',
            'half_sack_1_player_id','half_sack_2_player_id',
        ]
    ]

    return {
        'team_weekly_data':team_weekly_data,
        'kick_points':fg_ep_data,
        'play_yardage':play_yard_data,
        'non_kick_points':td_data,
        'turnovers':turnover_data,
        'penalties':penalty_data,
        'defensive':defense_data,
    }

def run_save_import(input_arguments):

    input_year_list = utilities.gen_year_list(
        input_arguments['ingest_start_year'],
        input_arguments['ingest_end_year']
    )
    nfl_dfs = run_nfl_import( input_year_list )
    __write_raw_data__(nfl_dfs,get_raw_data_path(input_arguments['input_version']))

if __name__ == "__main__":
    input_arguments = __read_args__()
    run_save_import(input_arguments)
