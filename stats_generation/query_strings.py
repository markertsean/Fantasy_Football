# Start of query string, player id and game week
__ind_query_string_start = (
        " SELECT play_player.player_id as player_id, play_player.team, game.week "
)

# Portion of query string at the end
# Makes sure to only grab finished games,
# combines like data by player and week, thus computing for a game
__ind_query_string_end = ( 
         " AND game.finished    = TRUE "
        +" GROUP BY play_player.player_id, play_player.team, game.week "
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
    +",   SUM(play_player.passing_cmp_air_yds) as pass_air_yds "
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

    # 
    +",   SUM(play_player.kickret_yds) as return_yds "
    +",   SUM(play_player.kickret_tds) as return_tds "
    +",   SUM(play_player.kickret_touchback) as touchbacks "
)


# D full select portion of query
__d_full_select_query = (
     " SELECT play_player.team as team, game.week "

    # Scored on
    +",   SUM(play_player.defense_frec_tds) as fumb_tds "
    +",   SUM(play_player.defense_int_tds) as int_tds "
    +",   SUM(play_player.defense_misc_tds) as misc_tds "
    +",   SUM(play_player.defense_int) as int "
    +",   SUM(play_player.defense_frec) as fumb_rec "
    +",   SUM(play_player.defense_sk) as sacks "
    +",   SUM(play_player.defense_safe) as safeties "
    +",   SUM(play_player.defense_fgblk) as fg_blk "
    +",   SUM(play_player.defense_puntblk) as punt_blk "
    +",   SUM(play_player.defense_xpblk) as xp_blk "

    # Features
    +",   SUM(play_player.defense_ast) as assisted_tackles "
    +",   SUM(play_player.defense_tkl_primary) as tackles "
    +",   SUM(play_player.defense_ffum) as fumb_forced "
    +",   SUM(play_player.defense_frec_yds) as fumb_rec_yds "
    +",   SUM(play_player.defense_int_yds) as int_yds "
    +",   SUM(play_player.defense_misc_yds) as misc_yds "
    +",   SUM(play_player.defense_pass_def) as pass_def "
    +",   SUM(play_player.defense_qbhit) as qb_hit "
    +",   SUM(play_player.defense_sk_yds) as sack_yds "
    +",   SUM(play_player.defense_tkl_loss) as tackle_loss "
    +",   SUM(play_player.defense_tkl_loss_yds) as tkl_loss_yds "
)

__team_full_query_string = (
    # All team data that may be useful
     " SELECT play_player.team as team, game.week as week  "

    # Scored on
    +",   SUM(play_player.defense_ffum)         as def_fumb_forced "
    +",   SUM(play_player.defense_frec)         as def_fumb_rec "
    +",   SUM(play_player.defense_frec_yds)     as def_frec_yds "
    +",   SUM(play_player.defense_frec_tds)     as def_frec_tds "
    +",   SUM(play_player.defense_fgblk)        as def_fg_blk "
    +",   SUM(play_player.defense_xpblk)        as def_xp_blk "
    +",   SUM(play_player.defense_puntblk)      as def_punt_blk "
    +",   SUM(play_player.defense_int)          as def_int "
    +",   SUM(play_player.defense_int_yds)      as def_int_yds "
    +",   SUM(play_player.defense_int_tds)      as def_int_tds "
    +",   SUM(play_player.defense_misc_yds)     as def_misc_yds "
    +",   SUM(play_player.defense_misc_tds)     as def_misc_tds "
    +",   SUM(play_player.defense_pass_def)     as def_pass_def "
    +",   SUM(play_player.defense_qbhit)        as def_qbhit "
    +",   SUM(play_player.defense_sk)           as def_sack "
    +",   SUM(play_player.defense_sk_yds)       as def_sack_yds "
    +",   SUM(play_player.defense_safe)         as def_safety "
    +",   SUM(play_player.defense_tkl_primary)  as def_tkl "
    +",   SUM(play_player.defense_tkl)          as def_tkl_contributers "
    +",   SUM(play_player.defense_tkl_loss)     as def_tkl_loss "
    +",   SUM(play_player.defense_tkl_loss_yds) as def_tkl_loss_yds "

    +",   SUM(play_player.fumbles_forced)    as off_fumb_forced "
    +",   SUM(play_player.fumbles_notforced) as off_fumb_unforced "
    +",   SUM(play_player.fumbles_lost)      as off_fumb_lost "
    +",   SUM(play_player.fumbles_rec)       as off_fumb_rec "
    +",   SUM(play_player.fumbles_rec_tds)   as off_fumb_rec_tds "

    +",   SUM(play_player.punting_tot) as punts "
    +",   SUM(play_player.punting_yds) as punt_yds "
    +",   SUM(play_player.punting_blk) as punt_blk "
    +",   SUM(play_player.puntret_tot) as punt_rets "
    +",   SUM(play_player.puntret_yds) as punt_ret_yds "
    +",   SUM(play_player.puntret_tds) as punt_ret_tds "

    +",   SUM(play_player.kicking_tot)       as kickoffs "
    +",   SUM(play_player.kicking_all_yds)   as kickoff_all_yds "
    +",   SUM(play_player.kicking_i20)       as kickoff_in_20 "
    +",   SUM(play_player.kicking_rec)       as kickoff_own_recovery "
    +",   SUM(play_player.kicking_rec_tds)   as kickoff_own_recovery_tds "
    +",   SUM(play_player.kicking_touchback) as kickoff_touchback "

    +",   SUM(play_player.kicking_fga)       as fg_att "
    +",   SUM(play_player.kicking_fgb)       as fg_blk "
    +",   SUM(play_player.kicking_fgm)       as fg_made "
    +",   SUM(play_player.kicking_fgmissed)  as fg_miss "
    +",   SUM(play_player.kicking_fgm_yds)   as fg_yds "
    +",   MAX(play_player.kicking_fgm_yds)   as fg_yds_max " 
    +",   SUM(play_player.kicking_fgmissed_yds) as fg_miss_yds "
    +",   MIN(NULLIF(play_player.kicking_fgmissed_yds,0)) as fg_miss_yds_min "

    +",   SUM(play_player.kicking_xpa)       as xp_att "
    +",   SUM(play_player.kicking_xpb)       as xp_blk "
    +",   SUM(play_player.kicking_xpmade)    as xp_made "
    +",   SUM(play_player.kicking_xpmissed)  as xp_miss "
    +",   SUM(play_player.kickret_ret)       as kickrets"
    +",   SUM(play_player.kickret_yds)       as kickret_yds "
    +",   SUM(play_player.kickret_tds)       as kickret_tds "
    +",   SUM(play_player.kickret_touchback) as kickret_touchback "

    +",   SUM(play_player.passing_att)    as pass_att "
    +",   SUM(play_player.passing_cmp)    as pass_cmp "
    +",   SUM(play_player.passing_incmp)  as pass_incmp "
    +",   SUM(play_player.passing_int)    as pass_int "
    +",   SUM(play_player.passing_sk)     as pass_sack "
    +",   SUM(play_player.passing_sk_yds) as pass_sack_yds "
    +",   SUM(play_player.passing_tds)    as pass_tds "
    +",   SUM(play_player.passing_yds)    as pass_yds "

    +",   SUM(play_player.rushing_att)      as rush_att "
    +",   SUM(play_player.rushing_yds)      as rush_yds "
    +",   SUM(play_player.rushing_tds)      as rush_tds "
    +",   SUM(play_player.rushing_loss)     as rush_loss "
    +",   SUM(play_player.rushing_loss_yds) as rush_loss_yds "


    +" FROM game "
    +" JOIN play_player "
    +" ON game.gsis_id = play_player.gsis_id "
)

__team_end_query_string = (
     "  AND game.finished    = TRUE "

    +" GROUP BY play_player.team, game.week "
    +" ORDER BY play_player.team, game.week "
)