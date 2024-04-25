# Andrew Fisher, Theodore Schnieder, Henry Gray, Nick Kuranda
#
# Program to analyze the efficiency of NFL teams with their chosen run 
# percentage based on how far they made it into the playoffs. 

from playoffResults import PlayoffResults
import requests
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC

key_dict = {
"Baltimore Ravens" : "Baltimore",
"Baltimore" : "Baltimore",

"Cleveland Browns": "Cleveland",
"Cleveland": "Cleveland",

"Pittsburgh Steelers": "Pittsburgh",
"Pittsburgh": "Pittsburgh",

"Cincinnati Bengals": "Cincinnati",
"Cincinnati": "Cincinnati",

"Buffalo Bills": "Buffalo",
"Buffalo": "Buffalo",

"Miami Dolphins": "Miami",
"Miami": "Miami",

"New York Jets": "New York Jets",
"NY Jets": "New York Jets",

"New England Patriots": "New England",
"New England": "New England",

"Kansas City Chiefs": "Kansas City",
"Kansas City": "Kansas City",

"Oakland Raiders": "Las Vegas",
"Las Vegas": "Las Vegas",
"Las Vegas Raiders": "Las Vegas",

"Denver Broncos": "Denver",
"Denver": "Denver",

"San Diego Chargers": "Los Angeles Chargers",
"Los Angeles Chargers": "Los Angeles Chargers",
"LA Chargers": "Los Angeles Chargers",

"Houston Texans": "Houston",
"Houston": "Houston",

"Jacksonville Jaguars": "Jacksonville",
"Jacksonville": "Jacksonville",

"Tennessee Titans": "Tennessee",
"Tennessee": "Tennessee",

"Indianapolis Colts" : "Indianapolis",
"Indianapolis" : "Indianapolis",

"San Francisco 49ers": "San Francisco",
"San Francisco": "San Francisco",

"St. Louis Rams": "Los Angeles Rams",
"Los Angeles Rams": "Los Angeles Rams",
"LA Rams": "Los Angeles Rams",

"Arizona Cardinals": "Arizona",
"Arizona": "Arizona",

"Seattle Seahawks": "Seattle",
"Seattle": "Seattle",

"Dallas Cowboys": "Dallas",
"Dallas": "Dallas",

"Washington Football Team": "Washington",
"Washington Redskins": "Washington",
"Washington Commanders": "Washington",
"Washington": "Washington",

"Philadelphia Eagles": "Philadelphia",
"Philadelphia": "Philadelphia",

"New York Giants": "New York Giants",
"NY Giants": "New York Giants",

"Detroit Lions": "Detroit",
"Detroit": "Detroit",

"Chicago Bears": "Chicago",
"Chicago": "Chicago",

"Green Bay Packers": "Green Bay",
"Green Bay": "Green Bay",

"Minnesota Vikings": "Minnesota",
"Minnesota": "Minnesota",

"Carolina Panthers": "Carolina",
"Carolina": "Carolina",

"New Orleans Saints": "New Orleans",
"New Orleans": "New Orleans",

"Atlanta Falcons": "Atlanta",
"Atlanta": "Atlanta",

"Tampa Bay Buccaneers": "Tampa Bay",
"Tampa Bay": "Tampa Bay",

}

NUM_TEAMS = 32

def get_rushing_pcts():
    toReturn = {}
    for i in range (21):
        year = str(2024 - i)
        url = "https://www.teamrankings.com/nfl/stat/rushing-play-pct?date=" + year + "-03-1"

        
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        toReturn[int(year)] = get_one_year(soup, int(year))
    return toReturn


def get_passing_yards_per_attempt(sched_dict):
    
    toReturn = {}
    for i in range (21):
        year = str(2024 - i)
        url = "https://www.teamrankings.com/nfl/stat/opponent-yards-per-pass-attempt?date=" + year + "-03-1"

        
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        stats = get_one_year(soup, int(year))

        tmp = {}
        for team in stats.keys():
            opp_ypa = 0
            cnt = 0
            for opp in sched_dict[int(year)][team]:
                opp_ypa += float(stats[opp]['pct'])
                cnt += 1
            tmp[team] = round(opp_ypa / cnt, 4)
        for team in tmp.keys():
            stats[team]['pct'] = tmp[team]
        
       
        toReturn[int(year)] = stats
    return toReturn

def add_stats_to_dict(dic, stats_dict, stat):
    #assumes format to be a dictionary with years as keys and each year has every team with a stat called 'pct'
    for year in dic.keys():
        for team in dic[year].keys():
            dic[year][team][stat] = stats_dict[year][team]['pct']
    return dic


def get_one_year(soup, year):
    team_percents = {}

    schedule_table = soup.find(class_="tr-table datatable scrollable")

    rows = schedule_table.find_all('tr')

    for row in rows:
        name_location = row.find('td', {"class" : "text-left nowrap"})
        percent_location = row.find('td', {"class" : "text-right"})
        
        if name_location is not None and percent_location is not None:
            name = name_location["data-sort"]
            percent = percent_location["data-sort"]
            realName = key_dict[name]
            if year < 2020:
                team_percents[realName] = dict(pct = float(percent), outcome = (2 / (32 / 12)))
            else:
                 team_percents[realName] = dict(pct = float(percent), outcome = (2 / (32 / 14)))

    return team_percents

def get_opp_rushing_per_attempt(sched_dict):
    toReturn = {}
    for i in range (21):
        year = str(2024 - i)
        url = "https://www.teamrankings.com/nfl/stat/opponent-yards-per-rush-attempt?date=" + year + "-03-1"

        
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        stats = get_one_year(soup, int(year))

        tmp = {}
        for team in stats.keys():
            opp_ypa = 0
            cnt = 0
            for opp in sched_dict[int(year)][team]:
                opp_ypa += float(stats[opp]['pct'])
                cnt += 1
            tmp[team] = round(opp_ypa / cnt, 4)
        for team in tmp.keys():
            stats[team]['pct'] = tmp[team]
        
       
        toReturn[int(year)] = stats
    return toReturn


def make_sched_dict():
    team_schedules = {}
    for i in range (21):
        year = str(2023 - i)
        url = "https://www.pro-football-reference.com/years/" + year + "/games.htm"
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        year = int(year) + 1
        
        team_schedules[year] = {}
        schedule_table = soup.find(id="games")
        rows = schedule_table.find_all('tr')
        for row in rows:
            round_location = row.find('th')
            if not round_location.text.isnumeric():
                continue
            winner = key_dict[row.find('td', {"data-stat" : "winner"}).text]
            loser = key_dict[row.find('td', {"data-stat" : "loser"}).text]

            winner = key_dict[winner]
            loser = key_dict[loser]

            # if year == 2024:
            #     print(winner)

            if winner not in team_schedules[year]:
                team_schedules[year][winner] = []
            if loser not in team_schedules[year]:
                team_schedules[year][loser] = []

            team_schedules[year][winner].append(loser)
            team_schedules[year][loser].append(winner)
            
            
    return team_schedules


def plot(full_dict):
    x_rushpcts = []
    y_outcomes = []
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            x_rushpcts.append(abs (year_dict[team]['pct'] - .5))
            y_outcomes.append(year_dict[team]['outcome'])

    plt.scatter(x_rushpcts, y_outcomes)
    plt.plot(np.unique(x_rushpcts), np.poly1d(np.polyfit(x_rushpcts, y_outcomes, 1))(np.unique(x_rushpcts)))
    plt.show()

def train_classifier(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    return model

def make_feat_matrix(full_dict):
    outerList = []
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            innerList = []
            innerList.append(abs (year_dict[team]['pct'] - .5))
            innerList.append(year_dict[team]['opp_ypa'])
            innerList.append(year_dict[team]['opp_ypc'])

            outerList.append(innerList)

    return np.array(outerList)

def make_label_arr(full_dict):
    outerList = []
    # dict = {.75 : 1, .875 : 1, 2 : 2, 4 : 3, 8 : 4, 16 : 5, 32 : 6}
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            outerList.append(year_dict[team]['outcome'])
            
    return np.array(outerList)

def main():
    # Below are just other urls you could run this on
    # BEWARE the rate limited request
    full_dict = {}
    # sched_dict = make_sched_dict()
    # # KEYS ARE THE LAST YEAR OF THE SEASON E.G 2023-2024 is coded as 2024
    
    full_dict = get_rushing_pcts()
    playoffResultsObj = PlayoffResults(key_dict)
    full_dict = playoffResultsObj.get_playoff_results(full_dict)
        
    # full_dict = add_stats_to_dict(full_dict, get_passing_yards_per_attempt(sched_dict), "opp_ypa")
    # full_dict = add_stats_to_dict(full_dict, get_opp_rushing_per_attempt(sched_dict), "opp_ypc")


   
    # with open('data_dict', 'rb') as file:
    #     full_dict = pickle.load(file)

    print(full_dict)
    
    # with open('data_dict', 'wb') as fp:
    #     pickle.dump(full_dict, fp)
    #     print('dictionary saved successfully to file')

    # xs = make_feat_matrix(full_dict)
    # ys = make_label_arr(full_dict)
    # print(len(xs))
    # # print(len(ys))
    # model = train_classifier(xs[: 500], ys[: 500])
    
    #  # Evaluate the model on test data
    # y_test = model.predict(xs[500:])
    # print(y_test)
    #plot(full_dict)
    # with open('data_dict', 'wb') as fp:
    #     pickle.dump(full_dict, fp)
    #     print('dictionary saved successfully to file')
   

if __name__ == "__main__":
    main()
   