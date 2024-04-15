# Theodore Schneider and Nick Kuranda
#
# Scraper program to find the probability that a team wins
# given that they are leading yards per passing attempt at the half

import requests
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

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

# get_player_name
# Parameters: Soup object
# Purpose: Extracts and returns the team name from the provided BeautifulSoup 
#          object.
# def get_player_name(soup):
    
#     player_name_location = soup.find(id="info")
#     player_name_container = player_name_location.find('h1')

#     player_name = player_name_container.find("span").text.strip()
#     return player_name
    

# get_game_urls
# Parameters: The main url to parse
# Purpose: "Scrapes the partial URL of every box score in the team schedule and 
#           returns a list of full URLs.
# Returns a list of partial urls
# def get_game_urls(url):
#     page = requests.get(url)
#     soup = BeautifulSoup(page.content, "html.parser")

#     schedule_table = soup.find(id="last5")
#     rows = schedule_table.find_all('tr')

#     game_urls = []

#     for row in rows:
#         dates = row.find_all('th', {"data-stat" : "date"})
#         if not dates:
#             continue

#         link_element = dates[0].find('a', href=True)

#         if link_element:
#             game_href = link_element['href']
#             url_root = re.match(r'.*\.com', url)[0]
#             game_url = f'{url_root}{game_href}'
#             game_urls.append(game_url)

#     return game_urls

# scrape_play_details
# Parameters: A player name and a url of a game
# Purpose: To get the passing information from a game up to the half
# Returns: True if the player's team led in passing yards per attempt
#          at halftime
# def scrape_play_details(game_url, player_name):
#     total_passing_attempts = 0
#     total_passing_yards = 0
#     total_passing_attempts_other = 0
#     total_passing_yards_other = 0

#     # """Scrapes and prints play details for each game."""
#     game_page = requests.get(game_url)
#     game_soup = BeautifulSoup(game_page.text.replace('<!--', '').replace('-->', ''), 'html.parser')

#     play_table = game_soup.find(id="pbp")

#     if play_table:
#         rows = play_table.find_all('tr')

#         for row in rows:
#             details = row.find_all('td', {"data-stat": "detail"})
#             if not details:
#                 continue

#             detail_text = details[0].text.strip().lower()

#             if player_name.lower() in detail_text and 'pass' in detail_text:
#                 # Extract yards from the detail_text using a regular expression
#                 yards_match = re.search(r'(-?\d+)', detail_text)
#                 if yards_match:
#                     passing_yards_str = yards_match.group(1)

#                     # Convert to integer, considering the negative sign
#                     passing_yards = int(passing_yards_str)

                
#                     # Increment passing attempts counter and add passing yards
#                     total_passing_attempts += 1
#                     total_passing_yards += passing_yards

#                     # NEEED TO RESET PASSING YARDS AND PASSING ATTEMPTS EACH GAME
#             elif player_name.lower() not in detail_text and 'pass' in detail_text:
#                 yards_match = re.search(r'(-?\d+)', detail_text)
#                 if yards_match:
#                     passing_yards_str = yards_match.group(1)

#                     # Convert to integer, considering the negative sign
#                     passing_yards = int(passing_yards_str)
                
#                     # Increment passing attempts counter and add passing yards
#                     total_passing_attempts_other += 1
#                     total_passing_yards_other += passing_yards



#             quarter = row.find_all('th', {"data-stat": "quarter"})
#             quarter_num = quarter[0].text.strip()

#             if quarter_num > "2":
#                 break
        
#         if total_passing_yards/total_passing_attempts > total_passing_yards_other/ total_passing_attempts_other:
#             return True 
#         else:
#             return False 
#     else:
#         print("Warning: 'pbp' element not found on the page.")
#         return False 
    
# get_team_name
# Purpose: Gets a team name from a website
# def get_team_name(soup):
#     # """Extracts and returns the team name from the provided BeautifulSoup object."""
#     team_location = soup.find(id="info")
#     team_paragraphs = team_location.find_all('p')

#     for team_info in team_paragraphs:
#         if team_info.find('strong', string='Team'):
#             team_name_location = team_info.find('a')
#             if team_name_location:
#                 return team_name_location.text.strip()

#     return None
def get_rushing_pcts():
    toReturn = {}
    for i in range (21):
        year = str(2024 - i)
        url = "https://www.teamrankings.com/nfl/stat/rushing-play-pct?date=" + year + "-03-1"

        
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        toReturn[int(year)] = get_one_year(soup)
    return toReturn


def get_one_year(soup):
    team_percents = {}
    #print(soup)
    schedule_table = soup.find(class_="tr-table datatable scrollable")

    rows = schedule_table.find_all('tr')

    for row in rows:
        name_location = row.find('td', {"class" : "text-left nowrap"})
        percent_location = row.find('td', {"class" : "text-right"})
        
        if name_location is not None and percent_location is not None:
            name = name_location["data-sort"]
            percent = percent_location["data-sort"]
            realName = key_dict[name]
            team_percents[realName] = dict(pct = percent, outcome = 0)

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
                    teams[year][realName]['outcome'] = 2
                else: 
                    teams[year][realName]['outcome'] = 1



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

def get_div_champ_results_for_year(row, year, teams):
    cols = row.find_all("td")
    
    team1 = cols[1].find('a').text.lstrip()
    teams[year][key_dict[team1]]['outcome'] = 4
    
    if cols[1].find('b') != None:
        teams[year][key_dict[team1]]['outcome'] = 5


    team2 = cols[3].find('a').text.lstrip()
    teams[year][key_dict[team2]]['outcome'] = 3
    # score = cols[2]
    # match = re.search(r'(\d+)\D+(\d+)', score.text)
    # print(match.group(1))
    # score_outcomes(year, teams, team1, team2, [int(match.group(1)), 
                                            #    int(match.group(2))])

    team3 = cols[5].find('a').text.lstrip()
    teams[year][key_dict[team3]]['outcome'] = 4
    
    if cols[5].find('b') != None:
        teams[year][key_dict[team3]]['outcome'] = 5


    team4 = cols[7].find('a').text.lstrip()
    teams[year][key_dict[team4]]['outcome'] = 3




    # score2 = cols[6]
    # scores2 = score2.text.split('-')
    # match2 = re.search(r'(\d+)\D+(\d+)', score2.text)
    # score_outcomes(year, teams, team3, team4, [int(match2.group(1)), 
                                            #    int(match2.group(2))])
def plot(full_dict):
    tuple_list = []
    for year in full_dict:
        year_dict = full_dict[year]
        for team in year_dict:
            tuple_list.append((abs (year_dict[team]['pct'] - .5), year_dict[team]['outcome']))

    plt.scatter(*zip(*tuple_list))
    plt.show()


def main():
    # Below are just other urls you could run this on
    # BEWARE the rate limited request
    full_dict = {}
    # KEYS ARE THE LAST YEAR OF THE SEASON E.G 2023-2024 is coded as 2024
    full_dict = get_rushing_pcts()
        
    # print (full_dict)

    
    results_url = "https://en.wikipedia.org/wiki/NFL_playoff_results"
    wild_card, divisional, conference = get_tables(results_url)
    wild_card_divisional_points(wild_card, full_dict, False)
    wild_card_divisional_points(divisional, full_dict, True)
    conference_superBowl_points(conference, full_dict)
    # print (full_dict[2024])
    plot(full_dict)
#     team_name   = get_team_name(soup)

#     if player_name and team_name:
#         print(f"Player Name: {player_name}")
#         print(f"Team Name: {team_name}")
#     else:
#         print("Player Name Not Found.")

#     # Get and print the game URLs
#     game_urls = get_game_urls(url)
  
#     # Scrape and print play details for each game
    
#     i = 0
#     temp_list = []
#     for game_url in game_urls:
#         leading_ypa = scrape_play_details(game_url, player_name)
        
        
#         if leading_ypa:
#             temp_list.append(i)
#         i = i+1

#     # Calculate probabilties
#     prob_b_given_a, prob_win = get_stat_given_win(temp_list, url)

#     prob_b_given_not_a = 1 - prob_b_given_a

#     prob_lose = 1 - prob_win

#     # Bayes theorem
#     prob = (prob_win*prob_b_given_a)/ ((prob_win*prob_b_given_a) + prob_lose*prob_b_given_not_a)

#     final_prob = prob * 100
#     final_prob = round(float(final_prob), 2)

#     print("The probability that the " + team_name + " win given that they lead in yards per passing attempt at the half is " + str(final_prob) + "%")
   

if __name__ == "__main__":
    main()
