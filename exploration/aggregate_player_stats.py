import numpy as np
import pandas as pd

from imp import load_source

query_strings = load_source('query_strings', '../stats_generation/query_strings.py' )
gps = load_source(  'generate_player_stats', '../stats_generation/generate_player_stats.py' )



def generate_kicker_features( end_year, n_weeks=4, start_year=2009 ):
    
    all_kicker_data = pd.DataFrame()

    # Get all the team preseason data
    # Can id by team, week, year
    for year in range( start_year, end_year ):
        new_frame = gps.generate_stats( 'K', year, season_type='Preseason' )
        new_frame['year'] = year
        all_kicker_data = pd.concat( [all_kicker_data, new_frame], ignore_index=True )
        
    # Preseason weeks make -4 to 0
    all_kicker_data['week'] = all_kicker_data['week']-4

    # Get all the Kicker regular season data
    # Can id by team, week, year
    for year in range( start_year, end_year ):
        new_frame = gps.generate_stats( 'K', year )
        new_frame['year'] = year
        all_kicker_data = pd.concat( [all_kicker_data, new_frame], ignore_index=True )

    # Ignore some team stuff, can get from joining with team
    all_kicker_data = all_kicker_data.drop( ['opp_team','home_flag','away_flag','xp_made','xp_miss'],axis=1 )

    # Generate previous rolling sum
    prev_kick = calc_prev_player_stats( all_kicker_data, ['fg_made','fg_miss','fg_made_yds','fg_miss_yds','fg_made_max'] )

    # Combine present values with rolling sums
    all_kicker_data = pd.merge( all_kicker_data, prev_kick, on=['player_id','team','year','week'] )

    # Drop all the preseason stuff
    all_kicker_data = all_kicker_data.loc[ all_kicker_data['week']>0 ]

    # Note if the data includes preseason stuff
    # If the first four games, flag as preseason data included
    # This is tricky, as can have a bye-week
    # Therefore, group things, find the first n_weeks, and flag those as 1
    inds = all_kicker_data.groupby(['player_id','year'], as_index=False).nth( range(0, n_weeks) ).index.values

    all_kicker_data    [       'few_reg_weeks'] = 0
    all_kicker_data.loc[ inds, 'few_reg_weeks'] = 1
    
    k_team_comb = all_kicker_data

    new_features = k_team_comb[['player_id','team','year','week','few_reg_weeks']].copy()

    wk_str = str(n_weeks)
    # Engineer some new features, mostly normalizing other features by kick/game
    new_features['fg_made'           ] = k_team_comb['fg_made'].copy()
    new_features['fg_made_prev_'+wk_str+'_avg'] = k_team_comb['fg_made_prev_'+wk_str] / n_weeks
    new_features['fg_acc_prev_'+wk_str        ] = k_team_comb['fg_made_prev_'+wk_str] / ( k_team_comb['fg_made_prev_'+wk_str] + 
                                                                                          k_team_comb['fg_miss_prev_'+wk_str] )

    new_features['fg_made_yds_prev_'+wk_str+'_avg'] = k_team_comb['fg_made_yds_prev_'+wk_str] / k_team_comb['fg_made_prev_'+wk_str]
    new_features['fg_miss_yds_prev_'+wk_str+'_avg'] = k_team_comb['fg_miss_yds_prev_'+wk_str] / k_team_comb['fg_miss_prev_'+wk_str]

    max_group_k = k_team_comb.groupby(['player_id','year'],as_index=False).rolling(100,min_periods=1).max()
    max_group_k['week'  ] = max_group_k['week'].astype(int)
    max_group_k['fg_max'] = max_group_k['fg_made_max']
    new_features['fg_max_season'] = pd.merge( k_team_comb, max_group_k, on=['player_id','year','week'] )['fg_max']
    max_group_k = 0

    new_features['fg_max_m_avg'  ] = new_features['fg_max_season'                  ] - new_features['fg_made_yds_prev_'+wk_str+'_avg']
    new_features['fg_made_m_miss'] = new_features['fg_made_yds_prev_'+wk_str+'_avg'] - new_features['fg_miss_yds_prev_'+wk_str+'_avg']
    
    return new_features

def generate_full_team_aggregate( end_year, n_weeks=4, start_year=2009, drop_preseason=True ):

    # Generate aggregate of team statistics from preseason and regular season
    team_stats_df = aggregate_pre_reg_team_stats( end_year )

    # Get sums from previous games
    prev_team     = calc_prev_team_stats( team_stats_df, team_stats_df.columns.values[4:], n_weeks )

    # Combine the present values with the recent averages
    team_stats_df = pd.merge( team_stats_df, prev_team, on=['team','year','week'] )

    # Drop all the preseason stuff, unless the user wants to keep it
    # If user doesn't drop, need to include a larger range for indicating preseason
    pre_mod = 0
    if (drop_preseason):
        team_stats_df = team_stats_df.loc[ team_stats_df['week']>0 ]
        pre_mod = 4
        
    # Note if the data includes preseason stuff
    # If the first four games, flag as preseason data included
    # This is tricky, as can have a bye-week
    # Therefore, group things, find the first n_weeks, and flag those as 1
    inds = team_stats_df.groupby(['team','year'], as_index=False).nth( range(0,n_weeks+pre_mod) ).index.values

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
    stat_list = stat_list + ['kickoffs','punts','fg_miss','kickret_tds','punt_ret_tds']

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

    # Defensive turnovers, just take a few
    stat_list = stat_list + ['def_fumb_rec','def_fumb_forced','def_int']

    # Pressure
    stat_list = stat_list + ['def_tkl_loss','def_sack','def_pass_def']
    
    # Return all of the 
    return all_team_data[stat_list]


# Calculates the sum of previous statistics
# User provides the frame and columns to sum over,
#   as well as the number of weeks,
# Returned frame is sorted by team and year,
def calc_prev_player_stats( 
                            inp      ,    # Dataframe containing preseason/regular season stuff 
                            use_cols ,    # Column names to do sums for
                            n_wk=4   ,    # Number of weeks to perform calculation. Def 4 ( num of preseason games )
                            avg_cols=None # Optional list of columns to perform averaging on
                           ):
    
    return calc_prev_stats( 
                    inp      ,    # Dataframe containing preseason/regular season stuff 
                    use_cols ,    # Column names to do sums for
                    'player_id',    # Whether team or player
                    n_wk     ,    # Number of weeks to perform calculation. Def 4 ( num of preseason games )
                    avg_cols      # Optional list of columns to perform averaging on
                   )

        
    assert ( ( 'player_id' in inp.columns ) & 
             ( 'week' in inp.columns ) & 
             ( 'year' in inp.columns ) & 
             ( 'team' in inp.columns ) ), "calc_prev_player_stats input dataframe requires ['player_id','week','year','team'] columns"
        
    # Make sure the df is sorted
    inp_df = inp.sort_values( ['player_id','year','week'] ).copy()
    
    # Generate new column names, use_cols + _prev_ndays
    new_cols = [str(col) + '_prev_' + str(n_wk) for col in use_cols]

    # Make sure we include week, for indexing purposes
    if ( type( use_cols ) == np.ndarray ):
        use_cols = use_cols.tolist()
    
    # Output frame,
    #  just set up columns and indexes,
    #  will return these columns
    new_frame = pd.DataFrame( index=inp_df.index, columns=new_cols )

    foo = ( inp_df.groupby(['player_id','year'], 
                           as_index=False, 
                           group_keys=False )
                           [use_cols]
                           .rolling( n_wk )
                           .sum()
                           .shift(1) )

    foo.columns = new_cols

    foo.index = inp_df.index

    bar =(inp_df.sort_values( ['player_id','year','week'] )
                        [['player_id','team','year','week']] )

    bar.index = inp_df.index

    foo['player_id'] = bar['player_id']
    foo['team'] = bar['team']
    foo['year'] = bar['year']
    foo['week'] = bar['week']
    
    return foo.copy()



# Calculates the sum of previous statistics
# User provides the frame and columns to sum over,
#   as well as the number of weeks,
# Returned frame is sorted by team and year,
def calc_prev_team_stats( 
                    inp      ,    # Dataframe containing preseason/regular season stuff 
                    use_cols ,    # Column names to do sums for
                    n_wk=4   ,    # Number of weeks to perform calculation. Def 4 ( num of preseason games )
                    avg_cols=None # Optional list of columns to perform averaging on
                        ):
     
    return calc_prev_stats( 
                    inp      ,    # Dataframe containing preseason/regular season stuff 
                    use_cols ,    # Column names to do sums for
                    'team'   ,    # Whether team or player
                    n_wk     ,    # Number of weeks to perform calculation. Def 4 ( num of preseason games )
                    avg_cols      # Optional list of columns to perform averaging on
                   )
        
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
                           .rolling( n_wk, min_periods=1 )
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

# Calculates the sum of previous statistics
# User provides the frame and columns to sum over,
#   as well as the number of weeks,
# Can also provide averages
def calc_prev_stats( 
                    inp      ,    # Dataframe containing preseason/regular season stuff 
                    use_cols ,    # Column names to do sums for
                    t_p      ,    # Whether team or player
                    n_wk=4   ,    # Number of weeks to perform calculation. Def 4 ( num of preseason games )
                    avg_cols=None # Optional list of columns to perform averaging on
                   ):
    
    assert ( 
              (  (t_p == 'team'      ) & ( 'team'      in inp.columns ) ) |
              (  (t_p == 'player_id' ) & ( 'player_id' in inp.columns ) )
           ), 't_p must be "team" or "player_id", and must be present in input frame'
    
    assert ( ( 'week' in inp.columns ) & 
             ( 'year' in inp.columns ) ), "calc_prev_stats input dataframe requires ['week','year'] columns"

    
    assert ( ( type(avg_cols) == None        ) |
             ( isinstance(avg_cols,np.ndarray) ) |
             ( isinstance(avg_cols,list      ) ) ), "avg_cols must be of type None, list, or np.ndarray "
        
    # Make sure the df is sorted
    inp_df = inp.sort_values( [t_p,'year','week'] ).copy()
    
    # Generate new column names, use_cols + _prev_ndays
    new_cols = [str(col) + '_prev_' + str(n_wk) for col in use_cols]

    # Make sure we include week, for indexing purposes
    if ( type( use_cols ) == np.ndarray ):
        use_cols = use_cols.tolist()
    
    # Output frame,
    #  just set up columns and indexes,
    #  will return these columns
    new_frame = pd.DataFrame( index=inp_df.index, columns=new_cols )

    foo = ( inp_df.groupby([t_p,'year'], 
                           as_index=False, 
                           group_keys=False )
                           [use_cols]
                           .rolling( n_wk, min_periods=0 )
                           .sum()
                           .shift(1) )
    
    # If averaging, get the number of weeks summed,
    #  and divide the sum by number of weeks summed
    if ( type(avg_cols) != None ):
        
        # Number of items in a summation
        n = ( inp_df.groupby([t_p,'year'], 
                           as_index=False, 
                           group_keys=False )
                           [use_cols]
                           .rolling( n_wk, min_periods=0 )
                           .count()
                           .shift(1)
            )

        # Keep track of new column list
        avg_col_list = []
        
        for col in avg_cols:
            avg_col_list.append( col+'_avg_'+str(n_wk) )
            foo[ avg_col_list[-1] ] = foo[col] / n[col]

        # Reset the column names using the new names & avgs
        foo.columns = np.append( new_cols, avg_col_list )

    else:
        # Reset the column names to the new names
        foo.columns = new_cols
        
    # Match the index so we can join back together
    foo.index = inp_df.index

    bar =(inp_df.sort_values( [t_p,'year'] )
                        [[t_p,'year','week']] )

    bar.index = inp_df.index

    foo[t_p   ] = bar[t_p   ]
    foo['year'] = bar['year']
    foo['week'] = bar['week']
    
    return foo.copy()