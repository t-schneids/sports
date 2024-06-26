# Andrew Fisher, Theodore Schnieder, Henry Gray, Nick Kuranda
#
# Program to analyze the efficiency of NFL teams with their chosen run 
# percentage based on how far they made it into the playoffs. 

from playoffResults import PlayoffResults
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from models import *
import pickle



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
                team_percents[realName] = dict(pct = float(percent), outcome = 1)
            else:
                 team_percents[realName] = dict(pct = float(percent), outcome = 1)

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

            if winner not in team_schedules[year]:
                team_schedules[year][winner] = []
            if loser not in team_schedules[year]:
                team_schedules[year][loser] = []

            team_schedules[year][winner].append(loser)
            team_schedules[year][loser].append(winner)
            
            
    return team_schedules



def main():
    # Below are just other urls you could run this on
    # BEWARE the rate limited request
    full_dict = {}
    # sched_dict = make_sched_dict()
    # # KEYS ARE THE LAST YEAR OF THE SEASON E.G 2023-2024 is coded as 2024
    
    # full_dict = get_rushing_pcts()
    # playoffResultsObj = PlayoffResults(key_dict)
    # full_dict = playoffResultsObj.get_playoff_results(full_dict)
        
    # full_dict = add_stats_to_dict(full_dict, get_passing_yards_per_attempt(sched_dict), "opp_ypa")
    # full_dict = add_stats_to_dict(full_dict, get_opp_rushing_per_attempt(sched_dict), "opp_ypc")

    with open('data_dict', 'rb') as file:
        full_dict = pickle.load(file)


    # print(full_dict)
    
    # with open('data_dict', 'wb') as fp:
    #     pickle.dump(full_dict, fp)
    #     print('dictionary saved successfully to file')

    xs = make_feat_matrix(full_dict)
    ys = make_label_arr(full_dict)



    years_list = [year for year in range(2003, 2024)]
    linear_results = train_test_eval_linear(xs, ys)
    trees_results = train_test_eval_trees(xs, ys)

    # plt.scatter(years_list, linear_results, color='orange', label="Linear Model")
    # plt.xlabel("Year")
    # plt.title("Error for linear model")
    # plt.ylabel("Average error in predicted outcome")
    # plt.xticks(years_list, rotation='vertical')
    # # plt.legend()
    # plt.show()

    # plt.clf()
    # plt.scatter(years_list, trees_results, color='blue', label="Random Forest Classifier")
    # plt.xlabel("Year")
    # plt.title("Error for random forest model")
    # plt.ylabel("Number of playoff teams correctly predicted")
    # plt.xticks(years_list, rotation='vertical')
    # plt.show()

    # plt.clf()
    # plt.scatter(years_list, linear_results, color='orange', label="Linear Model")
    # plt.scatter(years_list, trees_results, color='blue', label="Random Forest Classifier")
    # plt.xlabel("Year")
    # plt.title("Error Comparison")
    # plt.ylabel("Average error in predicted outcome")
    # plt.xticks(years_list, rotation='vertical')
    # plt.legend()
    # plt.show()

    plt.clf()
    plt.scatter(years_list, trees_results, color='blue', label="Random Forest Classifier")
    plt.xlabel("Year")
    plt.title("Playoff prediction true positives")
    plt.ylabel("Number of playoff teams correctly predicted")
    plt.xticks(years_list, rotation='vertical')
    plt.show()



if __name__ == "__main__":
    main()
   