import pandas as pd

from sqlalchemy import create_engine

from types import *

import datetime

import query_strings



# Current time and date
__now = datetime.datetime.now()


# Default engine path, for me, Sean Markert
__engine_path = 'postgresql://nfldb:football@localhost/nfldb'


# Strings below to form SQL queries
# Left in seperate file, as they are messy as hell

# Start of query string, player id and game week
__ind_query_string_start = query_strings.__ind_query_string_start

# Portion of query string at the end
# Makes sure to only grab finished games,
# combines like data by player and week, thus computing for a game
__ind_query_string_end = query_strings.__ind_query_string_end

# Portion of the query string that joins the tables
__ind_query_string_join = query_strings.__ind_query_string_join

# Fumble strings, some of fumble stats is for scoring
# the rest can be used for statistics
__ind_query_string_fumb = query_strings.__ind_query_string_fumb


# All the features and predictions for kickers
__ind_query_string_k = query_strings.__ind_query_string_k

# QB query items
__ind_query_string_qb_score = query_strings.__ind_query_string_qb_score
__ind_query_string_qb_feat  = query_strings.__ind_query_string_qb_feat

# WR query items
__ind_query_string_wr_score = query_strings.__ind_query_string_wr_score
__ind_query_string_wr_feat  = query_strings.__ind_query_string_wr_feat

# TE query items
__ind_query_string_te_score = query_strings.__ind_query_string_te_score
__ind_query_string_te_feat  = query_strings.__ind_query_string_te_feat

# RB query items
__ind_query_string_rb_score = query_strings.__ind_query_string_rb_score
__ind_query_string_rb_feat  = query_strings.__ind_query_string_rb_feat

# Team query items
__team_full_query_string = query_strings.__team_full_query_string
__team_end_query_string  = query_strings.__team_end_query_string






    
    
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

# Generates certain averages from values in the table
# Also generates touchdown totals
def __gen_team_avgs( df ):
    df['kickoff_all_yds_avg' ] = df['kickoff_all_yds' ] / df['kickoffs'    ]
    df['def_tkl_loss_yds_avg'] = df['def_tkl_loss_yds'] / df['def_tkl_loss']
    df['kickret_yds_avg'     ] = df['kickret_yds'     ] / df['kickrets'    ]
    df['puntret_yds_avg'     ] = df['punt_ret_yds'    ] / df['punts'    ]
    td_cols = filter(lambda x: '_tds' in x, df.columns.values)
    df['tds'] = df[td_cols].aggregate('sum',axis=1)
    return df





# If your path to the SQL database is different 
# than the default
# Format: <db engine>://<db user name>:<password>@localhost/<db name>
def set_engine_path( inp_str ):
    assert type( inp_str ) == StringType, 'Set engine path requires an input string'
    global __engine_path
    __engine_path = inp_str




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




# All the possibly useful statistics for teams, by weeks
# Will generate some probably useless data, but the user
#   can sort out what they want to use
def generate_team_stats( inp_year, season_type='Regular' ):
    return generate_stats( 'Team', inp_year, season_type )

    
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
    return generate_stats( 'K', inp_year, season_type )


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
    return generate_stats( 'QB', inp_year, season_type )


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
    return generate_stats( 'WR', inp_year, season_type )


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
    return generate_stats( 'TE', inp_year, season_type )


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
    return generate_stats( 'RB', inp_year, season_type )
    

    
# Combination of above functions
# Generates the dataset for the query
# Query strings exist in query_strings.py
# Can edit the queries there
#
# Individual players are built by position,
#   First selection of id and week,
#   then items pertaining to the scores,
#   then features that may prove useful,
#   finally the joining of the tables
#   with conditional selection,
#   and grouping
#
# Team is built very differently,
#   and thus will return seperate
#   from the other query building
#
def generate_stats( position, inp_year, season_type='Regular' ):
    
    # Make sure we are querying the right positions
    assert ((position=='K' ) or 
            (position=='QB') or 
            (position=='WR') or 
            (position=='TE') or 
            (position=='RB') or
            (position=='Team') ), 'Position must be one of "K", "QB", "WR", "TE", "RB", or "Team" '
        
    # Checks that inp_year and season_type valid
    # Returns the query strings to select that year and season
    year_str, season_str = __run_assertions( inp_year, season_type )
    
    
    # Set our engine path for querying
    engine = create_engine( __engine_path )

    
    # Start the string, and build it from the position of the player
    # Basically player id and week
    query_str = __ind_query_string_start

    
    # Team is seperate from the others, 
    #   just handle seperate case here
    #   and return
    if ( position == 'Team' ):

        query_str = (
            __team_full_query_string
            +year_str
            +season_str
            +__team_end_query_string
        )

        df = pd.read_sql_query(query_str,con=engine)
        return __gen_team_avgs( df )

    
    # Otherwise, add to the query string
    # whatever is specific to that position
    elif ( position == 'K'  ):
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
    
    # Add the bottom stuff if individual, 
    #   joining, selecting years/seasons/positions
    query_str = query_str + (
         __ind_query_string_join
        +year_str
        +season_str
        +"  AND player.position  = '%s' " % position
        +__ind_query_string_end
    )
    
    return pd.read_sql_query(query_str,con=engine)