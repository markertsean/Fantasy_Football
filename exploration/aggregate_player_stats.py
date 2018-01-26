import numpy as np
import pandas as pd

from imp import load_source

query_strings = load_source('query_strings', '../stats_generation/query_strings.py' )
gps = load_source(  'generate_player_stats', '../stats_generation/generate_player_stats.py' )


        
# Generates the features for input player type
# Mostly previous performance
def generate_player_features( end_year, position, n_weeks=4, start_year=2009 ):

    # Def can be passed in different ways
    if ( 
        ( position== 'D') | 
        ( position== 'd') | 
        ( position== 'def') | 
        ( position== 'Def') |
        ( position== 'DEF') 
       ):
        return generate_def_features( end_year, n_weeks, start_year )
    
    assert ( 
            ( position == 'QB' ) |
            ( position == 'WR' ) |
            ( position == 'TE' ) |
            ( position == 'RB' ) |
            ( position ==  'K' ) 
           ), "Position must be 'QB', 'RB', 'WR', 'TE', 'K', or 'DEF'"

    # List of features, vary for each position
    #  so just put in another function for clarity
    agg_stuff, opp_agg, team_stuff, reorg_stuff, ret_feats_head = __return_feature_string_arrays(position)
    
    
    # Rename some parameters
    min_year = start_year
    max_year =   end_year
    wk_str   = str(n_weeks)

    
    # For the Player SQL query
    all_player_data = pd.DataFrame()

    
    # Get all the preseason data
    # Can id by team, week, year
    for year in range( min_year, max_year+1 ):
        new_frame = gps.generate_stats( position, year, season_type='Preseason' )
        new_frame['year'] = year
        all_player_data = pd.concat( [all_player_data, new_frame], ignore_index=True )

    # Rebrand preseason week
    all_player_data['week'] = all_player_data['week']-4

    # Get all the player regular season data
    # Can id by team, week, year
    for year in range( min_year, max_year+1 ):
        new_frame = gps.generate_stats( position, year )
        new_frame['year'] = year
        all_player_data = pd.concat( [all_player_data, new_frame], ignore_index=True )

        
        
    # Ignore some team stuff, can get from joining with team
    all_player_data = all_player_data.drop( ['opp_team','home_flag','away_flag'],axis=1 )
    
    
    # Rolling sums from previous games
    prev_player = calc_prev_player_stats( all_player_data, agg_stuff, n_weeks )
    
    
    # Combine present values with rolling sums
    all_player_data = pd.merge( all_player_data, prev_player, on=['player_id','year','week'] )

    
    # Drop all the preseason stuff
    all_player_data = all_player_data.loc[ all_player_data['week']>0 ]

    
    # Note if the data includes preseason stuff
    # If the first four games, flag as preseason data included
    # This is tricky, as can have a bye-week
    # Therefore, group things, find the first n_weeks, and flag those as 1
    inds = all_player_data.groupby(['player_id','year'], as_index=False).nth( range(0, n_weeks) ).index.values

    all_player_data    [       'few_reg_weeks'] = 0
    all_player_data.loc[ inds, 'few_reg_weeks'] = 1
    
    
    # Generate team stats
    team_stats_df = generate_full_team_aggregate( end_year, n_weeks, start_year, drop_preseason=False )

    
    # Opposition team stuff to grab
    # May have low/hi stats due to tough/weak teams
    opp_df = calc_opp_avg( team_stats_df, opp_agg, n_weeks )

    
    # Stuff to grab from team df
    team_stuff = [ item+wk_str for item in team_stuff ]

    
    
    # Combine the player data with team data
    temp_frame = pd.merge( 
                           all_player_data, 
                           team_stats_df[['team','week','year']+team_stuff], 
                           on=['team','week','year'],
                           suffixes=('','_team')
                         )
    
    # For some reason att written as attempts
    if ( 'pass_attempts_prev_'+wk_str in temp_frame.columns.values ):
        temp_frame['pass_att_prev_'+wk_str] = temp_frame['pass_attempts_prev_'+wk_str]

    
    # Stuff to grab in the returned frame, organized a little
    reorg_stuff = [ item+wk_str for item in reorg_stuff ]
    
    
    # Re-organize the frame
    new_frame = temp_frame[
                            ret_feats_head +
                            reorg_stuff
                          ]
    
    # Stuff to rename from the team frame
    team_rn_dict = dict( zip( team_stuff, ['team_'+item for item in team_stuff] ) )

    
    # Rename some more rows
    new_frame = new_frame.rename( index=str, columns=team_rn_dict )
        
    # Combine with the opposition frame
    new_frame = pd.merge( new_frame, 
                          opp_df   ,
                          on=['team','week','year'],
                          how='left'
                        )
    
    return new_frame




# Generates the defensive features
# Mostly reliant of team stats, including
#  how previous teams performed against
#  this team in the past
def generate_def_features( end_year, n_weeks=4, start_year=2009 ):


    wk_str  = str(n_weeks)

    team_stats_df = generate_full_team_aggregate( end_year,
                                                      n_weeks,
                                                      start_year,
                                                      drop_preseason=False )

    keep_list = ['team','week','year','includes_preseason',
                 'opp_team','opp_score','tds',
                 'rush_yds','pass_yds','fg_made']

    def_df = team_stats_df[ keep_list ].copy()

    # All scored the same, hard to predict individually
    #  but can likely predict def scores as a whole
    def_df['all_def_tds'] = team_stats_df[[
                                            'def_int_tds',
                                            'def_frec_tds',
                                            'def_misc_tds',
                                            'kickret_tds',
                                            'punt_ret_tds'
                                          ]].sum(axis=1)

    # Same thing with turnovers
    def_df['all_def_turn'] = team_stats_df[[
                                            'def_fumb_rec',
                                            'def_int'
                                           ]].sum(axis=1)

    # Extremely rare
    def_df['def_safety'] = team_stats_df['def_safety']

    # Can take this
    def_df['def_sack'  ] = team_stats_df['def_sack'  ]


    def_agg = ['opp_score_prev_4','home_flag_prev_4','away_flag_prev_4','pass_sack_prev_4']
    opp_agg = ['score_prev_4','tds_prev_4','rush_yds_prev_4','pass_yds_prev_4','fg_made_prev_4']

    def_df[def_agg] = team_stats_df[def_agg].copy()
    def_df['allowed_points_prev_4'] = def_df['opp_score_prev_4']
    def_df = def_df.drop( 'opp_score_prev_4', axis=1 )

    def_df[opp_agg] = team_stats_df[opp_agg].copy()
    def_df.rename(columns=dict(zip( opp_agg, ['allowed_'+x for x in opp_agg] )), inplace=True)


    # Aggregate some of the above

    # Prob good indicators
    def_df['all_def_tds_prev_4'] = team_stats_df[[
                                            'def_int_tds_prev_4',
                                            'def_frec_tds_prev_4',
                                            'def_misc_tds_prev_4',
                                            'kickret_tds_prev_4',
                                            'punt_ret_tds_prev_4'
                                          ]].sum(axis=1)

    def_df['all_def_turn_prev_4'] = team_stats_df[[
                                            'def_fumb_rec_prev_4',
                                            'def_int_prev_4'
                                           ]].sum(axis=1)

    def_df['def_safety_prev_4'] = team_stats_df['def_safety_prev_4']
    def_df['def_sack_prev_4'  ] = team_stats_df['def_sack_prev_4'  ]




    # Current 'allowed_' are for the current team and one game, 
    #  need to turn these into averages for past 4,
    #  and then average over previous four opponents
    other_team_list = ['allowed_tds_prev_4', 'allowed_rush_yds_prev_4', 'allowed_pass_yds_prev_4', 'allowed_fg_made_prev_4']

    foo = def_df[ ['team','opp_team','week','year'] ].copy()

    foo['opp_avg'+wk_str+'_tds'     ] = def_df['allowed_tds_prev_'     +wk_str]
    foo['opp_avg'+wk_str+'_rush_yds'] = def_df['allowed_rush_yds_prev_'+wk_str] 
    foo['opp_avg'+wk_str+'_pass_yds'] = def_df['allowed_pass_yds_prev_'+wk_str] 
    foo['opp_avg'+wk_str+'_fg_made' ] = def_df['allowed_fg_made_prev_' +wk_str] 


    # Foo will contain averages of an individual team's  
    #  values over previous games, for different opponents
    foo =(calc_prev_team_stats( foo, 
                                foo.columns.values[4:], 
                                avg_cols=foo.columns.values[4:] )
                              [
                                [
                                    'team',
                                    'week',
                                    'year',
                                    'opp_avg'+wk_str+'_tds_avg_'     +wk_str,
                                    'opp_avg'+wk_str+'_rush_yds_avg_'+wk_str,
                                    'opp_avg'+wk_str+'_pass_yds_avg_'+wk_str,
                                    'opp_avg'+wk_str+'_fg_made_avg_' +wk_str
                                ]
                              ]
        )

    # Rename foo columns
    foo.columns =  [
                        'team',
                        'week',
                        'year',
                        'opp_avg'+wk_str+'_tds',
                        'opp_avg'+wk_str+'_rush_yds',
                        'opp_avg'+wk_str+'_pass_yds',
                        'opp_avg'+wk_str+'_fg_made'
                   ]


    # Need to average foo over teams faced,
    #  so consider teams faced, join team with opp team
    #  then avg the opp team over past 4 games

    # Do the joining on opposing team
    bar =(pd.merge( def_df[['team','opp_team','week','year']], 
                    foo, 
                    left_on =['opp_team','week','year'], 
                    right_on=[    'team','week','year'])
                    .drop( 'team_y', axis=1 )
                    .rename( index=str, columns={'team_x':'team'} )
         )

    # Perform aggregation, so sums opposing team allowed yardage
    #  for a given team
    bar = calc_prev_team_stats( bar, bar.columns.values[4:] )



    use_df = def_df[['team','week','year','includes_preseason']].copy()

    # Targets
    for col in ['all_def_tds', 
                'all_def_turn', 
                'def_safety', 
                'def_sack']:
        use_df[col] = def_df[col].copy()

    # Opposing team's score
    use_df['allowed_points'] = def_df['opp_score']



    # Features
    use_df['home_frac_prev_4'] = def_df['home_flag_prev_4'] / ( def_df['home_flag_prev_4'] + 0.0 + def_df['away_flag_prev_4'] )


    # Aggregate features
    for col in ['all_def_tds_prev_4',
                'all_def_turn_prev_4',
                'def_safety_prev_4',
                'def_sack_prev_4',
                'allowed_points_prev_4']:
        use_df[col] = def_df[col].copy()

    # Combine with what the defenses allowed in the previous games
    #  against opposing teams, on average
    use_df = pd.merge( 
                        use_df, 
                        bar,
                        on=['team','year','week']
                     )

    # Only select regular season
    use_df = use_df.loc[ use_df['week']>0 ]
    
    return use_df.copy()


def generate_kicker_features( end_year, n_weeks=4, start_year=2009 ):
    
    all_kicker_data = pd.DataFrame()

    # Get all the team preseason data
    # Can id by team, week, year
    for year in range( start_year, end_year+1 ):
        new_frame = gps.generate_stats( 'K', year, season_type='Preseason' )
        new_frame['year'] = year
        all_kicker_data = pd.concat( [all_kicker_data, new_frame], ignore_index=True )
        
    # Preseason weeks make -4 to 0
    all_kicker_data['week'] = all_kicker_data['week']-4

    # Get all the Kicker regular season data
    # Can id by team, week, year
    for year in range( start_year, end_year+1 ):
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


# Generates the defensive features
# Mostly reliant of team stats, including
#  how previous teams performed against
#  this team in the past
def generate_def_features( end_year, n_weeks=4, start_year=2009 ):


    wk_str  = str(n_weeks)

    team_stats_df = generate_full_team_aggregate( end_year,
                                                      n_weeks,
                                                      start_year,
                                                      drop_preseason=False )

    keep_list = ['team','week','year','includes_preseason',
                 'opp_team','opp_score','tds',
                 'rush_yds','pass_yds','fg_made']

    def_df = team_stats_df[ keep_list ].copy()

    # All scored the same, hard to predict individually
    #  but can likely predict def scores as a whole
    def_df['all_def_tds'] = team_stats_df[[
                                            'def_int_tds',
                                            'def_frec_tds',
                                            'def_misc_tds',
                                            'kickret_tds',
                                            'punt_ret_tds'
                                          ]].sum(axis=1)

    # Same thing with turnovers
    def_df['all_def_turn'] = team_stats_df[[
                                            'def_fumb_rec',
                                            'def_int'
                                           ]].sum(axis=1)

    # Extremely rare
    def_df['def_safety'] = team_stats_df['def_safety']

    def_df['def_sack'  ] = team_stats_df['def_sack'  ]


    def_agg = ['opp_score_prev_4','home_flag_prev_4','away_flag_prev_4','pass_sack_prev_4']
    opp_agg = ['score_prev_4','tds_prev_4','rush_yds_prev_4','pass_yds_prev_4','fg_made_prev_4']

    def_df[def_agg] = team_stats_df[def_agg].copy()
    def_df['allowed_points_prev_4'] = def_df['opp_score_prev_4']
    def_df = def_df.drop( 'opp_score_prev_4', axis=1 )

    def_df[opp_agg] = team_stats_df[opp_agg].copy()
    def_df.rename(columns=dict(zip( opp_agg, ['allowed_'+x for x in opp_agg] )), inplace=True)


    # Aggregate some of the above

    # Prob good indicators
    def_df['all_def_tds_prev_4'] = team_stats_df[[
                                            'def_int_tds_prev_4',
                                            'def_frec_tds_prev_4',
                                            'def_misc_tds_prev_4',
                                            'kickret_tds_prev_4',
                                            'punt_ret_tds_prev_4'
                                          ]].sum(axis=1)

    def_df['all_def_turn_prev_4'] = team_stats_df[[
                                            'def_fumb_rec_prev_4',
                                            'def_int_prev_4'
                                           ]].sum(axis=1)

    def_df['def_safety_prev_4'] = team_stats_df['def_safety_prev_4']
    def_df['def_sack_prev_4'  ] = team_stats_df['def_sack_prev_4'  ]




    # Current 'allowed_' are for the current team and one game, 
    #  need to turn these into averages for past 4,
    #  and then average over previous four opponents



    other_team_list = ['allowed_tds_prev_4', 'allowed_rush_yds_prev_4', 'allowed_pass_yds_prev_4', 'allowed_fg_made_prev_4']

    foo = def_df[ ['team','opp_team','week','year'] ].copy()

    foo['opp_avg'+wk_str+'_tds'     ] = def_df['allowed_tds_prev_'     +wk_str]
    foo['opp_avg'+wk_str+'_rush_yds'] = def_df['allowed_rush_yds_prev_'+wk_str] 
    foo['opp_avg'+wk_str+'_pass_yds'] = def_df['allowed_pass_yds_prev_'+wk_str] 
    foo['opp_avg'+wk_str+'_fg_made' ] = def_df['allowed_fg_made_prev_' +wk_str] 


    # Foo will contain averages of an individual team's  
    #  values over previous games, for different opponents
    foo =(calc_prev_team_stats( foo, 
                                foo.columns.values[4:], 
                                avg_cols=foo.columns.values[4:] )
                              [
                                [
                                    'team',
                                    'week',
                                    'year',
                                    'opp_avg'+wk_str+'_tds_avg_'     +wk_str,
                                    'opp_avg'+wk_str+'_rush_yds_avg_'+wk_str,
                                    'opp_avg'+wk_str+'_pass_yds_avg_'+wk_str,
                                    'opp_avg'+wk_str+'_fg_made_avg_' +wk_str
                                ]
                              ]
        )

    # Rename foo columns
    foo.columns =  [
                        'team',
                        'week',
                        'year',
                        'opp_avg'+wk_str+'_tds',
                        'opp_avg'+wk_str+'_rush_yds',
                        'opp_avg'+wk_str+'_pass_yds',
                        'opp_avg'+wk_str+'_fg_made'
                   ]


    # Need to average foo over teams faced,
    #  so consider teams faced, join team with opp team
    #  then avg the opp team over past 4 games

    # Do the joining on opposing team
    bar =(pd.merge( def_df[['team','opp_team','week','year']], 
                    foo, 
                    left_on =['opp_team','week','year'], 
                    right_on=[    'team','week','year'])
                    .drop( 'team_y', axis=1 )
                    .rename( index=str, columns={'team_x':'team'} )
         )

    # Perform aggregation, so sums opposing team allowed yardage
    #  for a given team
    bar = calc_prev_team_stats( bar, bar.columns.values[4:] )



    use_df = def_df[['team','week','year','includes_preseason']].copy()

    # Targets
    for col in ['all_def_tds', 
                'all_def_turn', 
                'def_safety', 
                'def_sack']:
        use_df[col] = def_df[col].copy()

    # Opposing team's score
    use_df['allowed_points'] = def_df['opp_score']



    # Features
    use_df['home_frac_prev_4'] = def_df['home_flag_prev_4'] / ( def_df['home_flag_prev_4'] + 0.0 + def_df['away_flag_prev_4'] )


    # Aggregate features
    for col in ['all_def_tds_prev_4',
                'all_def_turn_prev_4',
                'def_safety_prev_4',
                'def_sack_prev_4',
                'allowed_points_prev_4']:
        use_df[col] = def_df[col].copy()

    # Combine with what the defenses allowed in the previous games
    #  against opposing teams, on average
    use_df = pd.merge( 
                        use_df, 
                        bar,
                        on=['team','year','week']
                     )

    # Only select regular season
    use_df = use_df.loc[ use_df['week']>0 ]
    
    return use_df.copy()


def generate_full_team_aggregate( end_year, n_weeks=4, start_year=2009, drop_preseason=True ):

    # Generate aggregate of team statistics from preseason and regular season
    team_stats_df = aggregate_pre_reg_team_stats( end_year )

    # Get sums from previous games
    prev_team     = calc_prev_team_stats( team_stats_df, team_stats_df.columns.values[4:], n_weeks )

    # Combine the present values with the recent averages
    team_stats_df = pd.merge( team_stats_df, prev_team, on=['team','year','week'] )

    # Drop all the preseason stuff, unless the user wants to keep it
    # If user doesn't drop, need to include a larger range for indicating preseason
    pre_mod = 4
    if (drop_preseason):
        team_stats_df = team_stats_df.loc[ team_stats_df['week']>0 ]
        pre_mod = 0
        
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
    for year in range( start_year, end_year+1 ):
        new_frame = gps.generate_stats( 'Team', year )
        new_frame['year'] = year
        reg_team_data = pd.concat( [reg_team_data, new_frame], ignore_index=True )

    # Pre-season
    pre_team_data = pd.DataFrame()
    # Get all the team data
    # Can id by team, week, year
    for year in range( start_year, end_year+1 ):
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

    assert ( ( avg_cols is None      ) |
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
    if ( avg_cols is not None ):
        
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


# Generates stats for opposing teams
# Takes average performance for opposing teams
#   over previous n_wk games,
#   serves as metric of how good that team is
# Then performs aggregate summing of the values
#   for previous n_wk for a given team
def calc_opp_avg( inp_df, use_cols, n_wks=4 ):


    # Make sure use_cols is actually in the array
    assert np.isin( use_cols, inp_df.columns.values ).all(), \
        'Elements '+str(use_cols)+' is not in input columns'
    
    
    twyo = ['team','week','year','opp_team']
    
    wk_str = str(n_wks)
    
    # Copy the frame with only what we need
    foo_df = inp_df[twyo+use_cols].copy()
    foo_df.rename(columns=dict(zip( use_cols, ['opp_avg_'+x+'_prev_'+wk_str for x in use_cols] )), inplace=True)

    new_cols = foo_df.columns.values[4:]

    # Foo will contain averages of an individual team's  
    #  values over previous games, for different opponents
    foo_avg =(    calc_prev_team_stats( foo_df, 
                                        new_cols, 
                                        avg_cols=new_cols )
                                       [
                                           twyo[:-1]+
                                           [ col+'_avg_'+wk_str for col in new_cols ]
                                       ]
             )


    # Need to average foo over teams faced,
    #  so consider teams faced, join team with opp team
    #  then avg the opp team over past 4 games

    # Do the joining on opposing team
    bar =(
            pd.merge( 
                        foo_df[['team','opp_team','week','year']], 
                        foo_avg, 
                        left_on =['opp_team','week','year'], 
                        right_on=[    'team','week','year'])
                        .drop( 'team_y', axis=1 )
                        .rename( index=str, columns={'team_x':'team'} 
                    )
         )

    # Perform aggregation, so sums opposing team allowed yardage
    #  for a given team
    bar = calc_prev_team_stats( bar, bar.columns.values[4:] )

    # Rename stuff because names are messy
    r_dict = {}
    for i in range(0,bar.columns.values[:-3].shape[0]):
        r_dict[ bar.columns.values[i] ] = bar.columns.values[i][:-13]
        
    return bar.rename( index=str, columns=r_dict ).copy()






# Put all the arrays of what to grab in one place
def __return_feature_string_arrays( position ):
    
    agg_stuff   = []
    opp_agg     = []
    team_stuff  = []
    reorg_stuff = []
    ret_feats   = []
    
    # This is pretty much the same for all positions
    opp_agg = ['tds','fg_made','rush_yds','pass_yds','def_tkl_loss','def_sack','def_pass_def']

    # Diff for qb
    team_stuff = ['tds_prev_','fg_made_prev_','fg_miss_prev_','home_flag_prev_','away_flag_prev_','kickoffs_prev_',
              'punts_prev_','rush_att_prev_','pass_att_prev_','rush_yds_prev_','pass_yds_prev_',]

    ret_feats_head = ['player_id','team','week','year','rush_yds', 'rush_tds',
                      'rec_yds', 'rec_tds', 'receptions','fumb_lost','few_reg_weeks',]
                           
    if ( position=='QB' ):
    
        agg_stuff = ['pass_yds', 'pass_tds', 'pass_int', 'rush_yds', 'rush_tds', 'rush_att','fumb_lost','fumb_rec_tds','fumb_rec',
                     'fumb_forced','fumb_nforced','pass_attempts', 'pass_complete','pass_incomplete', 'pass_air_yds', 'pass_air_yds_max', 
                     'sacks', 'sack_yards']

        reorg_stuff = [
                'pass_complete_prev_', 'pass_incomplete_prev_' , 'pass_int_prev_', 'pass_air_yds_prev_' , 'pass_air_yds_max_prev_',
                'pass_yds_prev_', 'pass_tds_prev_', 'pass_att_prev_', 'rush_yds_prev_', 'rush_tds_prev_', 'rush_att_prev_',
                'fumb_lost_prev_'  , 'fumb_rec_prev_'    , 'fumb_rec_tds_prev_', 'fumb_forced_prev_', 'fumb_nforced_prev_',
                'sacks_prev_', 'sack_yards_prev_', 'home_flag_prev_', 'away_flag_prev_',
                'tds_prev_', 'fg_made_prev_', 'fg_miss_prev_','kickoffs_prev_', 'punts_prev_'
              ]

        team_stuff = ['tds_prev_','fg_made_prev_','fg_miss_prev_', 
               'home_flag_prev_','away_flag_prev_','kickoffs_prev_','punts_prev_']

        ret_feats_head = ['player_id','team','week','year','rush_yds','rush_tds',
                  'pass_yds','pass_tds','pass_int','fumb_lost','few_reg_weeks',]

    elif ( position=='RB' ):
        agg_stuff = ['receptions','rec_target','rec_yds','rec_tds','rush_att','rush_yds','rush_tds','fumb_lost','fumb_rec',
                     'fumb_rec_tds','fumb_forced','fumb_nforced','yards_after_compl','return_yds','return_tds','touchbacks']

        reorg_stuff = ['receptions_prev_', 'rec_target_prev_', 'rec_yds_prev_', 'rec_tds_prev_', 'yards_after_compl_prev_',
                       'rush_att_prev_', 'rush_yds_prev_', 'rush_tds_prev_','return_yds_prev_', 'return_tds_prev_', 'touchbacks_prev_',
                       'fumb_lost_prev_', 'fumb_rec_prev_', 'fumb_rec_tds_prev_', 'fumb_forced_prev_', 'fumb_nforced_prev_',]
        
    elif ( position=='TE' ):
        agg_stuff = ['receptions','rec_target','rec_yds','rec_tds','fumb_lost','fumb_rec',
                     'fumb_rec_tds','fumb_forced','fumb_nforced','yards_after_compl']

        reorg_stuff = ['receptions_prev_', 'rec_target_prev_', 'rec_yds_prev_', 'rec_tds_prev_', 'yards_after_compl_prev_',
                       'fumb_lost_prev_', 'fumb_rec_prev_', 'fumb_rec_tds_prev_', 'fumb_forced_prev_', 'fumb_nforced_prev_', ]

        ret_feats_head = ['player_id','team','week','year','receptions', 'rec_yds', 'rec_tds', 
                          'fumb_lost','fumb_rec_tds','few_reg_weeks',]

    elif ( position=='WR' ):
        

        agg_stuff = ['receptions','rec_target','rec_yds','rec_tds','fumb_lost','fumb_rec',
                     'fumb_rec_tds','fumb_forced','fumb_nforced','yards_after_compl','return_yds','return_tds','touchbacks']

        reorg_stuff = ['receptions_prev_', 'rec_target_prev_', 'rec_yds_prev_', 'rec_tds_prev_', 'yards_after_compl_prev_',
                       'return_yds_prev_', 'return_tds_prev_', 'touchbacks_prev_',
                       'fumb_lost_prev_', 'fumb_rec_prev_', 'fumb_rec_tds_prev_', 'fumb_forced_prev_', 'fumb_nforced_prev_',]

        ret_feats_head = ['player_id','team','week','year','receptions', 'rec_yds', 'rec_tds', 
                          'fumb_lost','fumb_rec_tds','few_reg_weeks',]

                       
    elif ( position=='K' ):
        agg_stuff = ['fg_made', 'fg_made_max','fg_made_yds', 'fg_miss', 'fg_miss_yds']
        
        reorg_stuff = ['fg_made_prev_','fg_miss_prev_', 'fg_made_yds_prev_', 'fg_miss_yds_prev_','fg_made_max_prev_',]
        
        ret_feats_head = ['player_id','team','week','year','few_reg_weeks','fg_made']

    else:
        print 'aggregate_player_stats.__return_feature_string_arrays recieved unknown arg for position: ',
        print position
        
    return agg_stuff, opp_agg, team_stuff, reorg_stuff, ret_feats_head
