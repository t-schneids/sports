# Authors: Theodore Schneider, Andrew Fisher, Henry Gray, Nick Kuranda
# Purpose: To provide a way to collect playoff results for every NFL team
#          for the past 21 seasons


from bs4 import BeautifulSoup
import requests


class PlayoffResults:
    def __init__(self, key_dict) -> None:
        """
        Must take in a key dictionary for all teams for team name consistency.
        """
        self.key_dict = key_dict

    def get_playoff_results(self, full_dict):
        results_url = "https://en.wikipedia.org/wiki/NFL_playoff_results"
        wild_card, divisional, conference = self._get_tables(results_url)
        self._wild_card_divisional_points(wild_card, full_dict, False)
        self._wild_card_divisional_points(divisional, full_dict, True)
        self._conference_superBowl_points(conference, full_dict)

        return full_dict

    def _get_tables(self, url):
        soup = BeautifulSoup(requests.get(url).content, "html.parser")

        tables = soup.find_all(class_ = "wikitable sortable")
        wild_card = tables[0]
        divisional = tables[1]
        conference = tables[2]
    
        return wild_card, divisional, conference
    
    def _wild_card_divisional_points(self, wild_card, teams, isDivisional):
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
                                self._get_wild_results_for_year(rows, i, 
                                        rowspanVal, year, teams, isDivisional)


    def _conference_superBowl_points(self, conference, teams):
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
                                self._get_div_champ_results_for_year(row, year, 
                                                                teams)

    
    # This gets wild card and divisional results for a year despite the name
    def _get_wild_results_for_year(self, rows, index, rowSpanVal, year, teams, 
                                   isDivisional):
        print(teams)
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
                    realName = self.key_dict[name]
                    if isDivisional:
                        teams[year][realName]['outcome'] = 4
                    
                    else: 
                        if year < 2020:
                            teams[year][realName]['outcome'] = 2 
                        else:
                            teams[year][realName]['outcome'] = 2


    def _get_div_champ_results_for_year(self, row, year, teams):
        cols = row.find_all("td")
        
        team1 = cols[1].find('a').text.lstrip()
        teams[year][self.key_dict[team1]]['outcome'] = 16
        
        # won the superbowl
        if cols[1].find('b') != None:
            teams[year][self.key_dict[team1]]['outcome'] = 32


        team2 = cols[3].find('a').text.lstrip()
        teams[year][self.key_dict[team2]]['outcome'] = 8

        team3 = cols[5].find('a').text.lstrip()
        teams[year][self.key_dict[team3]]['outcome'] = 16
        
        # won the superbowl
        if cols[5].find('b') != None:
            teams[year][self.key_dict[team3]]['outcome'] = 32


        team4 = cols[7].find('a').text.lstrip()
        teams[year][self.key_dict[team4]]['outcome'] = 8