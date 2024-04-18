# Andrew Fisher, Theodore Schnieder, Henry Gray, Nick Kuranda
#
# Program to analyze the efficiency of NFL teams with their chosen run 
# percentage based on how far they made it into the playoffs. 

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
        for ateam in sched_dict[2024].keys():
            print(ateam)

        tmp = {}
        for team in stats.keys():
            opp_ypa = 0
            cnt = 0
            print(year)
            for opp in sched_dict[int(year)][team]:
                opp_ypa += float(stats[year][opp]['pct'])
                cnt += 1
            tmp[team] = round(opp_ypa / cnt, 4)
        for team in tmp.keys():
            stats[year][team]['pct'] = tmp[team]
        
       
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



def get_tables(url):
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

    tables = soup.find_all(class_ = "wikitable sortable")
    wild_card = tables[0]
    divisional = tables[1]
    conference = tables[2]
   
    return wild_card, divisional, conference


# Get wild card and divisional points for the dictionary
def wild_card_divisional_points(wild_card, teams, isDivisional):
    rows = wild_card.find('tbody').find_all('tr')

# Only works because the rows without years have an a and then a b
    for i in range(len(rows)):
       row = rows[i]
       if row:
            yeartd = row.find('td')
            if yeartd:
                yearb = yeartd.find('b')
                if yearb:
                    year_location =  yearb.find('a')
                    if year_location:
                        year = year_location.text
                        if int(year[-2:]) > 24 or int(year[-2:]) < 4:
                            continue
                        year = 2000 + int(year[-2:])
                        rowspanVal = int(yeartd['rowspan'])
                        get_wild_results_for_year(rows, i, rowspanVal, year, teams, isDivisional)

# This gets wild card and divisional results for a year despite the name
def get_wild_results_for_year(rows, index, rowSpanVal, year, teams, 
                              isDivisional):
    for i in range (index, index + rowSpanVal):
        offset = 0 # used to figure out if we are in the first column
        row = rows[i]
        cols = row.find_all("td")
        if i == index:
            offset = 1

        # loop through each year and get each team from the table 
        for j in range (0 + offset, 7 + offset, 2):
            team_col = cols[j]
            name_location = team_col.find("a")
            if name_location:
                name = name_location.text.lstrip()
                realName = key_dict[name]
                if isDivisional:
                    teams[year][realName]['outcome'] = 4
                   
                else: 
                    if year < 2020:
                        teams[year][realName]['outcome'] = 2 
                    else:
                        teams[year][realName]['outcome'] = 2



def conference_superBowl_points(conference, teams):
    rows = conference.find('tbody').find_all('tr')

# Only works because the rows without years have an a and then a b
    for i in range(len(rows)):
       row = rows[i]
       if row:
            yeartd = row.find('td')
            if yeartd:
                yearb = yeartd.find('b')
                if yearb:
                    year_location =  yearb.find('a')
                    if year_location:
                        year = year_location.text
                        if int(year[-2:]) > 24 or int(year[-2:]) < 4:
                            continue
                        year = 2000 + int(year[-2:])
                        get_div_champ_results_for_year(row, year, teams)

# def score_outcomes(year, teams, team1, team2, scores):
#     rTeam1 = key_dict[team1]
#     rTeam2 = key_dict[team2]
#     if int(scores[0]) > int(scores[1]):
#         teams[year][rTeam1]['outcome'] = 8
#         teams[year][rTeam2]['outcome'] = 4
#     else: 
#         teams[year][rTeam1]['outcome'] = 4
#         teams[year][rTeam2]['outcome'] = 8
def make_sched_dict():
    team_schedules = {}
    for i in range (21):
        year = str(2024 - i)
        url = "https://www.pro-football-reference.com/years/" + year + "/games.htm"
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        year = int(year)
        
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
            if year == 2024:

                print(winner)

            if winner not in team_schedules[year]:
                team_schedules[year][winner] = []
            if loser not in team_schedules[year]:
                team_schedules[year][loser] = []

            team_schedules[year][winner].append(loser)
            team_schedules[year][loser].append(winner)
            
            
    return team_schedules


def get_div_champ_results_for_year(row, year, teams):
    cols = row.find_all("td")
    
    team1 = cols[1].find('a').text.lstrip()
    teams[year][key_dict[team1]]['outcome'] = 16
    
    # won the superbowl
    if cols[1].find('b') != None:
        teams[year][key_dict[team1]]['outcome'] = 32


    team2 = cols[3].find('a').text.lstrip()
    teams[year][key_dict[team2]]['outcome'] = 8
    # score = cols[2]
    # match = re.search(r'(\d+)\D+(\d+)', score.text)
    # print(match.group(1))
    # score_outcomes(year, teams, team1, team2, [int(match.group(1)), 
                                            #    int(match.group(2))])

    team3 = cols[5].find('a').text.lstrip()
    teams[year][key_dict[team3]]['outcome'] = 16
    
    # won the superbowl
    if cols[5].find('b') != None:
        teams[year][key_dict[team3]]['outcome'] = 32


    team4 = cols[7].find('a').text.lstrip()
    teams[year][key_dict[team4]]['outcome'] = 8




    # score2 = cols[6]
    # scores2 = score2.text.split('-')
    # match2 = re.search(r'(\d+)\D+(\d+)', score2.text)
    # score_outcomes(year, teams, team3, team4, [int(match2.group(1)), 
                                            #    int(match2.group(2))])
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
    """
    Train a support vector machine (SVM) classifier on the training set
    Returns:
    clf_out (SVM classifier): Trained SVM classifier
    """
    # clf_out = SVC(C=1, kernel="linear", random_state=0)
    # clf_out.fit(X_train_in, y_train_in)

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
            #innerList.append(year_dict[team]['outcome'])
            # add other features hear
            outerList.append(innerList)

    return np.array(outerList)

def make_label_arr(full_dict):
    outerList = []
    dict = {.75 : 1, .875 : 1, 2 : 2, 4 : 3, 8 : 4, 16 : 5, 32 : 6}
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            outerList.append(dict[year_dict[team]['outcome']])
            
    return np.array(outerList)

def get_opponent_rushing_yards_per_attempt():
    toReturn = {}
    for i in range(21):
        year = str(2024 - i)
        url = f"https://www.teamrankings.com/nfl/stat/opponent-yards-per-rush-attempt?date={year}-03-1"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        stats = get_one_year(soup, int(year))

        total = 0
        num_teams = len(stats)
        for team in stats.keys():
            total += float(stats[team]['pct'])

        for team in stats.keys():
            if num_teams > 1:  
                adjusted_avg = (total - float(stats[team]['pct'])) / (num_teams - 1)
            else:
                adjusted_avg = float(stats[team]['pct'])
            stats[team]['opp_rush_ya'] = round(adjusted_avg, 4)

        toReturn[int(year)] = stats
    return toReturn


def main():
    # Below are just other urls you could run this on
    # BEWARE the rate limited request
    full_dict = {}
    sched_dict = make_sched_dict()
    # KEYS ARE THE LAST YEAR OF THE SEASON E.G 2023-2024 is coded as 2024
    
    # full_dict = get_rushing_pcts()
    with open('data_dict', 'rb') as fp:
        full_dict = pickle.load(fp)
        
    full_dict = add_stats_to_dict(full_dict, get_passing_yards_per_attempt(sched_dict), "opp_ypa")
    full_dict = add_stats_to_dict(full_dict, get_opponent_rushing_yards_per_attempt(), "opp_rypa")

    # with open('data_dict', 'wb') as fp:
    #     pickle.dump(full_dict, fp)
    #     print('dictionary saved successfully to file')


    # results_url = "https://en.wikipedia.org/wiki/NFL_playoff_results"
    # wild_card, divisional, conference = get_tables(results_url)
    # wild_card_divisional_points(wild_card, full_dict, False)
    # wild_card_divisional_points(divisional, full_dict, True)
    # conference_superBowl_points(conference, full_dict)

    print (full_dict)

    # xs = make_feat_matrix(full_dict)
    # ys = make_label_arr(full_dict)
    # print(len(xs))
    # print(len(ys))
    # model = train_classifier(xs[: 500], ys[: 500])
    
     # Evaluate the model on test data
    # y_test = model.predict(xs[500:])
    # print(y_test)
    #plot(full_dict)
    # with open('data_dict', 'wb') as fp:
    #     pickle.dump(full_dict, fp)
    #     print('dictionary saved successfully to file')
   

if __name__ == "__main__":
    main()
   