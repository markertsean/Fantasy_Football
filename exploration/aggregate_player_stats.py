import numpy as np
import pandas as pd

from imp import load_source

query_strings = load_source('query_strings', '../stats_generation/query_strings.py' )
gps = load_source(  'generate_player_stats', '../stats_generation/generate_player_stats.py' )


def generate_full_team_aggregate( end_year, n_weeks=4, start_year=2009 ):

    # Generate aggregate of team statistics from preseason and regular season
    team_stats_df = aggregate_pre_reg_team_stats( end_year )

    # Get sums from previous games
    prev_team     = calc_prev_team_stats( team_stats_df, team_stats_df.columns.values[4:], n_weeks )

    # Combine the present values with the recent averages
    team_stats_df = pd.merge( team_stats_df, prev_team, on=['team','year','week'] )

    # Drop all the preseason stuff
    team_stats_df = team_stats_df.loc[ team_stats_df['week']>0 ]

    # Note if the data includes preseason stuff
    # If the first four games, flag as preseason data included
    # This is tricky, as can have a bye-week
    # Therefore, group things, find the first n_weeks, and flag those as 1
    inds = team_stats_df.groupby(['team','year'], as_index=False).nth( range(0,n_weeks) ).index.values

    team_stats_df    [       'includes_preseason'] = 0
    team_stats_df.loc[ inds, 'includes_preseason'] = 1

    return team_stats_df
    
# Aggregates team statistics from preseason/regular season
def aggregate_pre_reg_team_stats( end_year, 
                                  start_year = 2009 ):

    # Regular season
    reg_team_data = pd.DataFrame()
    # Get all the team data
    # Can id by team, week, year
    for year in range( start_year, end_year ):
        new_frame = gps.generate_stats( 'Team', year )
        new_frame['year'] = year
        reg_team_data = pd.concat( [reg_team_data, new_frame], ignore_index=True )

    # Pre-season
    pre_team_data = pd.DataFrame()
    # Get all the team data
    # Can id by team, week, year
    for year in range( start_year, end_year ):
        new_frame = gps.generate_stats( 'Team', year, 'Preseason' )
        new_frame['year'] = year
        pre_team_data = pd.concat( [pre_team_data, new_frame], ignore_index=True )
    pre_team_data['week'] = pre_team_data['week']-4

    # Combine the two data sets
    all_team_data = pd.concat( [pre_team_data,reg_team_data] )
        
    # List of things we will use
    # Start with meta game stuff
    stat_list = ['team','opp_team','week','year', 'score', 'opp_score', 'home_flag', 'away_flag','tds']

    # Offense

    # General offensive play breakdowns
    #  Have these features, plus make the commented ones
    stat_list = stat_list + ['rush_att','rush_yds','pass_att','pass_yds','pass_cmp','pass_sack']

    # Get info on kicks
    stat_list = stat_list + ['kickoffs','punts','fg_miss']

    # How well off keeps the ball
    stat_list = stat_list + ['off_fumb_tot']

    # Kickoffs that land far,
    stat_list = stat_list + ['kickoff_in_20','kickoff_touchback']

    # How good they are at putting in the kicker
    # And how far that is
    stat_list = stat_list + ['fg_yds','fg_made']

    # Defense

    # How many scores total
    stat_list = stat_list + ['def_int_tds','def_frec_tds','def_misc_tds','def_safety']

    # Can defense do a good job one on one, 
    #  or does whole D need to tackle one guy
    stat_list = stat_list + ['def_tkl','def_tkl_contributers']

    # Defensive turnovers, just take 2
    stat_list = stat_list + ['def_fumb_forced','def_int']

    # Pressure
    stat_list = stat_list + ['def_tkl_loss','def_sack','def_pass_def']
    
    # Return all of the 
    return all_team_data[stat_list]


# Calculates the sum of previous statistics
# User provides the frame and columns to sum over,
#   as well as the number of weeks,
# Returned frame is sorted by team and year,
def calc_prev_team_stats( 
                    inp      ,    # Dataframe containing preseason/regular season stuff 
                    use_cols ,    # Column names to do sums for
                    n_wk=4        # Number of weeks to perform calculation. Def 4 ( num of preseason games )
                   ):
     
    assert ( ( 'week' in inp.columns ) & 
             ( 'year' in inp.columns ) & 
             ( 'team' in inp.columns ) ), "calc_prec_stats input dataframe requires ['week','year','team'] columns"
        
    # Make sure the df is sorted
    inp_df = inp.sort_values( ['team','year','week'] ).copy()
    
    # Generate new column names, use_cols + _prev_ndays
    new_cols = [str(col) + '_prev_' + str(n_wk) for col in use_cols]

    # Make sure we include week, for indexing purposes
    if ( type( use_cols ) == np.ndarray ):
        use_cols = use_cols.tolist()
    
    # Output frame,
    #  just set up columns and indexes,
    #  will return these columns
    new_frame = pd.DataFrame( index=inp_df.index, columns=new_cols )

    foo = ( inp_df.groupby(['team','year'], 
                           as_index=False, 
                           group_keys=False )
                           [use_cols]
                           .rolling( n_wk )
                           .sum()
                           .shift(1) )

    foo.columns = new_cols

    foo.index = inp_df.index

    bar =(inp_df.sort_values( ['team','year'] )
                        [['team','year','week']] )

    bar.index = inp_df.index

    foo['team'] = bar['team']
    foo['year'] = bar['year']
    foo['week'] = bar['week']
    
    return foo.copy()