import sys
sys.path.insert(1, '../data/')
import teams

import pickle
import pandas

team_stats_dict_files = ['2013.pkl',
                         '2014.pkl',
                         '2015.pkl',
                         '2016.pkl',
                         '2017.pkl',
                         '2018.pkl',
                         '2019.pkl']

for file_name in team_stats_dict_files:
    team_stats_dict = pickle.load(open('../data/pergame/' + file_name, 'rb'))
    variation_coefficients_by_team = {}
    for team in teams.teams:
        if team not in team_stats_dict:
            if team == 'New Orleans Hornets':
                team = 'New Orleans Pelicans'
            elif team == 'Charlotte Bobcats':
                team = 'Charlotte Hornets'
        per_game_stats = team_stats_dict[team]
        variation_coefficients_by_team[team] = per_game_stats.std(axis=0) / per_game_stats.mean(axis=0)
    pickle.dump(variation_coefficients_by_team,
                open('../data/variation_coefficients/' + file_name, 'wb'))


