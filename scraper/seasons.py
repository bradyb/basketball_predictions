from sportsreference.nba.boxscore import Boxscores
import datetime

if __name__ == "__main__":
    boxscores = Boxscores(datetime.date(2018, 10, 16), datetime.date(2019, 6, 13))
    for _, day_of_games in boxscores.games.items():
        for game in day_of_games:
            print(game['boxscore'])
