import pandas as pd

from sqlalchemy import create_engine

from types import *

import datetime


# Current time and date
__now = datetime.datetime.now()


# Default engine path, for me, Sean Markert
__engine_path = 'postgresql://nfldb:football@localhost/nfldb'



# If your path to the SQL database is different 
# than the default
# Format: <db engine>://<db user name>:<password>@localhost/<db name>
def set_engine_path( inp_str ):
    assert type( inp_str ) == StringType, 'Set engine path requires an input string'
    global __engine_path
    __engine_path = inp_str
    
    
# Runs the assertions that the variables are good
# Used for each query
def __run_assertions( inp_year, season_type ):

    # Make sure everything runs well...
    assert type( inp_year ) == IntType, \
        'inp_year must be an int'
    
    assert ( inp_year > 2008     ) and \
           ( inp_year <=__now.year ), \
        'nfldb only contains information from 2009 to present, update the database using nfldb-update'
    
    assert ( (season_type=='Regular'   ) or 
             (season_type=='Preseason' ) or 
             (season_type=='Postseason') ), \
        'nfldb season type must be "Regular", "Preseason", or "Postseason" '

    # Set the query strings using our conditions
    year_str   = " WHERE game.season_year = %i " % inp_year
    season_str = "  AND game.season_type = '%s' " % season_type 
    
    return year_str, season_str



# Generates a df containing all player id's and player names
def generate_player_id_names():
    
    engine = create_engine( __engine_path )

    query_str = \
    (
    " SELECT play_player.player_id as player_id, player.full_name as full_name "   
    " FROM play_player "
    " JOIN player "
    " ON play_player.player_id = player.player_id"
    " ORDER BY play_player.player_id"
    )
    
    df = pd.read_sql_query(query_str,con=engine)
    return df.drop_duplicates()



# Strings below to form SQL queries

# Start of query string, player id and game week
__ind_query_string_start = (
        " SELECT play_player.player_id as player_id, game.week "
)

# Portion of query string at the end
# Makes sure to only grab finished games,
# combines like data by player and week, thus computing for a game
__ind_query_string_end = ( 
         " AND game.finished    = TRUE "
        +" GROUP BY play_player.player_id, game.week "
        +" ORDER BY play_player.player_id, game.week "
)

# Portion of the query string that joins the tables
__ind_query_string_join = (
     " FROM game "
    +" JOIN play_player "
    +" ON game.gsis_id = play_player.gsis_id "
    +" JOIN player"
    +" ON play_player.player_id = player.player_id"
)

# Fumble strings, some of fumble stats is for scoring
# the rest can be used for statistics
__ind_query_string_fumb = (
    # Fumble scores
     ",   SUM(play_player.fumbles_lost) as fumb_lost "
    +",   SUM(play_player.fumbles_rec_tds) as fumb_rec_tds "    

    # More fumbles
    +",   SUM(play_player.fumbles_rec) as fumb_rec "
    +",   SUM(play_player.fumbles_forced) as fumb_forced "    
    +",   SUM(play_player.fumbles_notforced) as fumb_nforced "
)


# All the features and predictions for kickers
__ind_query_string_k = (
    # Extra points
     ",   SUM(play_player.kicking_xpmade) as xp_made "
    +",   SUM(play_player.kicking_xpmissed) as xp_miss "

    # Field goals    
    +",   SUM(play_player.kicking_fgm) as fg_made "
    +",   SUM(play_player.kicking_fgmissed) as fg_miss "

    # Maximum range + minimum missed
    +",   MAX(play_player.kicking_fgm_yds) as fg_made_max "
    +",   MIN(NULLIF(play_player.kicking_fgmissed_yds,0)) as fg_miss_min "
)

# QB query items
__ind_query_string_qb_score = (
    # Scored on
     ",   SUM(play_player.passing_yds) as pass_yds "
    +",   SUM(play_player.passing_tds) as pass_tds "
    +",   SUM(play_player.passing_int) as pass_int "
    +",   SUM(play_player.rushing_yds) as rush_yds "
    +",   SUM(play_player.rushing_tds) as rush_tds "
)
__ind_query_string_qb_feat = (        
    # Possible features
     ",   SUM(play_player.passing_att) as pass_attempts "
    +",   SUM(play_player.passing_cmp) as pass_complete "
    +",   SUM(play_player.passing_incmp) as pass_incomplete "
    +",   AVG(play_player.passing_cmp_air_yds) as pass_air_yds_avg "
    +",   MAX(play_player.passing_cmp_air_yds) as pass_air_yds_max "
    +",   SUM(play_player.passing_sk) as sacks "
    +",   SUM(play_player.passing_sk_yds) as sack_yards "
    
    #
    +",   SUM(play_player.rushing_att) as rush_att "
)

# WR query items
__ind_query_string_wr_score = (
    # Scored on
     ",   SUM(play_player.receiving_rec) as receptions "
    +",   SUM(play_player.receiving_yds) as rec_yds "
)    
__ind_query_string_wr_feat = (
    # Possible features
     ",   SUM(play_player.receiving_tar) as rec_target "
    +",   SUM(play_player.receiving_yac_yds) as yards_after_compl "

    # 
    +",   AVG(play_player.kickret_yds) as return_yds_avg "
    +",   SUM(play_player.kickret_yds) as return_yds "
    +",   SUM(play_player.kickret_tds) as return_tds "
    +",   SUM(play_player.kickret_touchback) as touchbacks "
)

# TE query items
__ind_query_string_te_score = (
    # Scored on
     ",   SUM(play_player.receiving_rec) as receptions "
    +",   SUM(play_player.receiving_yds) as rec_yds "
)        
__ind_query_string_te_feat = (        
    # Possible features
     ",   SUM(play_player.receiving_tar) as rec_target "
    +",   SUM(play_player.receiving_yac_yds) as yards_after_compl "
)

# RB query items
__ind_query_string_rb_score = (
    # Scored on
     ",   SUM(play_player.receiving_rec) as receptions "
    +",   SUM(play_player.receiving_yds) as rec_yds "
    +",   SUM(play_player.receiving_tds) as rec_tds"
    +",   SUM(play_player.rushing_att) as rush_att"
    +",   SUM(play_player.rushing_yds) as rush_yds"
    +",   SUM(play_player.rushing_tds) as rush_tds"
)
__ind_query_string_rb_feat = (
    # Possible features
     ",   SUM(play_player.receiving_tar) as rec_target "
    +",   SUM(play_player.receiving_yac_yds) as yards_after_compl "
    +",   AVG(play_player.receiving_yac_yds) as yards_after_compl_avg "
    +",   AVG(play_player.receiving_yds) as rec_yds_avg "

    # 
    +",   AVG(play_player.kickret_yds) as return_yds_avg "
    +",   SUM(play_player.kickret_yds) as return_yds "
    +",   SUM(play_player.kickret_tds) as return_tds "
    +",   SUM(play_player.kickret_touchback) as touchbacks "
)



# Kickers
#
# Usually score off 
#      PAT, will predict as more of a team and assume PAT
#      FG < 50, will predict from team and fg stats
#      FG > 50, " "
#
# Stats taken are xp/fg made/miss, maximimum yardage made, minimum yardage missed
# 
def generate_k_stats( inp_year, season_type='Regular' ):

    # Checks that inp_year and season_type valid
    # Returns the query strings
    year_str, season_str = __run_assertions( inp_year, season_type )

    # Set our engine path for querying
    engine = create_engine( __engine_path )

    query_str = (
        __ind_query_string_start
        +__ind_query_string_k   # All the kicker stuff
        +__ind_query_string_join
        +year_str
        +season_str
        +"  AND player.position  = 'K' "
        +__ind_query_string_end
    )

    return pd.read_sql_query(query_str,con=engine)

# Quarterbacks
#
# Usually score off 
#      Passing yards
#      Passing touchdowns
#      Passing interceptions
#      Fumbles lost
#      Fumbles recovered for a td
#
# Additional stats taken are 
#      Fumbles recovered, forced, not forced
#      Pass attempts, complete, incomplete
#      Pass air yards avg/max
#      Sacked, sacked yardage
#      Rush attempts
# 
def generate_qb_stats( inp_year, season_type='Regular' ):
    
    # Checks that inp_year and season_type valid
    # Returns the query strings
    year_str, season_str = __run_assertions( inp_year, season_type )
    
    # Set our engine path for querying
    engine = create_engine( __engine_path )

    query_str = (
        __ind_query_string_start
        +__ind_query_string_qb_score
        +__ind_query_string_fumb
        +__ind_query_string_qb_feat
        +__ind_query_string_join
        +year_str
        +season_str
        +"  AND player.position  = 'QB' "
        +__ind_query_string_end
    )

    return pd.read_sql_query(query_str,con=engine)

# Wide reciever
#
# Usually score off 
#      Receptions
#      Reception yards
#      Passing interceptions
#      Fumbles lost
#      Fumbles recovered for a td
#
# Additional stats taken are 
#      Fumbles recovered, forced, not forced
#      Target for reception
#      Yards after completion
#      Return yards total/avg
#      Returns for touchdowns
#      Returns to touchbacks
# 
def generate_wr_stats( inp_year, season_type='Regular' ):

    # Checks that inp_year and season_type valid
    # Returns the query strings
    year_str, season_str = __run_assertions( inp_year, season_type )
    
    # Set our engine path for querying
    engine = create_engine( __engine_path )
    
    query_str = (
        __ind_query_string_start
        +__ind_query_string_wr_score
        +__ind_query_string_fumb
        +__ind_query_string_wr_feat
        +__ind_query_string_join    
        +year_str
        +season_str
        +"  AND player.position  = 'WR' "
        +__ind_query_string_end
    )
    
    return pd.read_sql_query(query_str,con=engine)

# Tight end
#
# Usually score off 
#      Receptions
#      Reception yards
#      Passing interceptions
#      Fumbles lost
#      Fumbles recovered for a td
#
# Additional stats taken are 
#      Fumbles recovered, forced, not forced
#      Target for reception
#      Yards after completion
# 
def generate_te_stats( inp_year, season_type='Regular' ):

    # Checks that inp_year and season_type valid
    # Returns the query strings
    year_str, season_str = __run_assertions( inp_year, season_type )
    
    # Set our engine path for querying
    engine = create_engine( __engine_path )
    
    query_str = (
        __ind_query_string_start
        +__ind_query_string_te_score
        +__ind_query_string_fumb
        +__ind_query_string_te_feat
        +__ind_query_string_join
        +year_str
        +season_str
        +"  AND player.position  = 'TE' "
        +__ind_query_string_end
    )   

    return pd.read_sql_query(query_str,con=engine)

# Running back
#
# Usually score off 
#      Receptions
#      Reception yards
#      Passing interceptions
#      Fumbles lost
#      Fumbles recovered for a td
#
# Additional stats taken are 
#      Fumbles recovered, forced, not forced
#      Target for reception
#      Yards after completion
# 
def generate_rb_stats( inp_year, season_type='Regular' ):

    # Checks that inp_year and season_type valid
    # Returns the query strings
    year_str, season_str = __run_assertions( inp_year, season_type )
    
    # Set our engine path for querying
    engine = create_engine( __engine_path )
    
    query_str = (
        __ind_query_string_start
        +__ind_query_string_rb_score
        +__ind_query_string_fumb
        +__ind_query_string_rb_feat
        +__ind_query_string_join
        +year_str
        +season_str
        +"  AND player.position  = 'RB' "
        +__ind_query_string_end
    )
    
    return pd.read_sql_query(query_str,con=engine)

# Combination of above functions
def generate_stats( position, inp_year, season_type='Regular' ):
    
    # Make sure we are querying the right positions
    assert ((position=='K' ) or 
            (position=='QB') or 
            (position=='WR') or 
            (position=='TE') or 
            (position=='RB') ), 'Position must be one of "K", "QB", "WR", "TE", "RB" '
        
    # Checks that inp_year and season_type valid
    # Returns the query strings
    year_str, season_str = __run_assertions( inp_year, season_type )
    
    # Set our engine path for querying
    engine = create_engine( __engine_path )

    # Start the string, and build it from the position of the player
    query_str = __ind_query_string_start

    if   ( position == 'K'  ):
        query_str = query_str + __ind_query_string_k
        
    elif ( position == 'QB' ):
        query_str = query_str + ( 
            __ind_query_string_qb_score + 
            __ind_query_string_fumb     + 
            __ind_query_string_qb_feat  )
        
    elif ( position == 'WR' ):
        query_str = query_str + ( 
            __ind_query_string_wr_score + 
            __ind_query_string_fumb     + 
            __ind_query_string_wr_feat  )
        
    elif ( position == 'TE' ):
        query_str = query_str + ( 
            __ind_query_string_te_score + 
            __ind_query_string_fumb     + 
            __ind_query_string_te_feat  )
        
    elif ( position == 'RB' ):
        query_str = query_str + ( 
            __ind_query_string_rb_score + 
            __ind_query_string_fumb     + 
            __ind_query_string_rb_feat  )
        
    else:
        print position,' not found'
        return
    
    # Add the bottom stuff, joining, selecting years/seasons/positions
    query_str = query_str + (
         __ind_query_string_join
        +year_str
        +season_str
        +"  AND player.position  = '%s' " % position
        +__ind_query_string_end
    )
    
    return pd.read_sql_query(query_str,con=engine)