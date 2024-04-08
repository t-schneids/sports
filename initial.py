# Theodore Schneider and Nick Kuranda
#
# Scraper program to find the probability that a team wins
# given that they are leading yards per passing attempt at the half

import requests
import re
from bs4 import BeautifulSoup

# get_player_name
# Parameters: Soup object
# Purpose: Extracts and returns the team name from the provided BeautifulSoup 
#          object.
# def get_player_name(soup):
    
#     player_name_location = soup.find(id="info")
#     player_name_container = player_name_location.find('h1')

#     player_name = player_name_container.find("span").text.strip()
#     return player_name


# get_stat_given_win
# Parameters: A list of rows indicating games where a team led in yards per 
#             passing attempt and a url
# Purpose: To find the probability of leading passing yards per attempt given 
#   that you win while also finding the probability that a team wins in general
def get_one_year(soup):
    team_names = {}
    #print(soup)
    schedule_table = soup.find(class_="tr-table datatable scrollable")

    rows = schedule_table.find_all('tr')

    for row in rows:
        name_location = row.find('td', {"class" : "text-left nowrap"})
        percent_location = row.find('td', {"class" : "text-right"})
        
        if name_location is not None and percent_location is not None:
            name = name_location["data-sort"]
            percent = percent_location["data-sort"]
            team_names[name] = percent

    return team_names
    

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

def main():
    # Below are just other urls you could run this on
    # BEWARE the rate limited request
    full_dict = {}
    # KEYS ARE THE LAST YEAR OF THE SEASON E.G 2023-2024 is coded as 2024
    for i in range (21):
        year = str(2024 - i)
        url = "https://www.teamrankings.com/nfl/stat/rushing-play-pct?date=" + year + "-03-1"

        # Get and print the team name
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        full_dict[year] = get_one_year(soup)
        
    print (full_dict)
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
