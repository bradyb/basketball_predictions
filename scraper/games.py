from sportsreference.nba.boxscore import Boxscore
import datetime
import pandas as pd

if __name__ == "__main__":
    years = ['2019']#, '2019']
    for year in years:
        file_path = year + '_games.txt'
        opened_file = open(file_path, 'r')
        line = opened_file.readline()
        season_dfs = []
        while line:
            boxscore = Boxscore(line.rstrip())
            season_dfs.append(boxscore.dataframe)
            line = opened_file.readline()
        pd.concat(season_dfs).to_pickle(year + '.pkl')
