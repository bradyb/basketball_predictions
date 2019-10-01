from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import pandas as pd

STATS_NAME_LIST = [
            "field_goal_percentage",
            "three_point_field_goal_percentage",
            "free_throw_attempts",
            "free_throw_percentage",
            "offensive_rebounds",
            "defensive_rebounds",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "personal_fouls",
            "points",
            "true_shooting_percentage",
            "effective_field_goal_percentage",
            "three_point_attempt_rate",
            "free_throw_attempt_rate",
            "offensive_rebound_percentage",
            "defensive_rebound_percentage",
            "assist_percentage",
            "steal_percentage",
            "block_percentage",
            "turnover_percentage",
        ]

HOME_NAME_LIST = ["home_" + stat_name for stat_name in STATS_NAME_LIST]
AWAY_NAME_LIST = ["away_" + stat_name for stat_name in STATS_NAME_LIST]

def add_game_for_team(team_stats_dict, game, stat_name_list, team_name):
    game_stats = game[stat_name_list]
    game_stats.columns = STATS_NAME_LIST
    if team_name in team_stats_dict:
        team_stats_dict[team_name] = team_stats_dict[team_name].append(game_stats)
    else:
        team_stats_dict[team_name] = game_stats

game_data_files_list = ['2013.pkl',
                        '2014.pkl',
                        '2015.pkl',
                        '2016.pkl',
                        '2017.pkl',
                        '2018.pkl',
                        '2019.pkl',]

for year_file in game_data_files_list:
    df = pickle.load(open('../data/' + year_file, 'rb'))
    team_stats_dict = {}

    for _, game in df.iterrows():
        if game['home_points'] > game['away_points']:
            home_name = game['winning_name']
            away_name = game['losing_name']
        else:
            home_name = game['losing_name']
            away_name = game['winning_name']
        add_game_for_team(team_stats_dict,
                          game.to_frame().transpose(),
                          HOME_NAME_LIST,
                          home_name)
        add_game_for_team(team_stats_dict,
                          game.to_frame().transpose(),
                          AWAY_NAME_LIST,
                          away_name)

    pickle.dump(team_stats_dict, open('../data/pergame/' + year_file, 'wb'))
