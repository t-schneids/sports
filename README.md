# Sports Analytics Final Project-
# Andrew Fisher, Teddy Schneider, Nick Kuranda, Henry Gray

# Purpose - to take a deep dive into NFL statistics to figure out which strategies take teams deep into the postseason.



# Dependencies: 

bs4
requests
sklearn
numpy
pickle
matplotlib.pyplot

playoffResults.py contains code to scrape and score playoff results for every
team for the past 21 years, adding each season-team's score to its stat
dictionary in the overall dictionary

prediction.py contains driver code and functions to scrape rushing percentages,
opponent teams defensive yards per carry, and opponent teams defensive yards per pass attempt,
adding all to the season-teams stat dictionary in the overall dictionary

models.py contains code used for our analysis including the original linear analysis
and analysis of linear and random forest models with added features

For speed purposes and to avoid rate-limited request errors, all of our 
scraped data is serialized with pickle and stored in data_dict. The commented
out function calls in main can be brought back in if live scraping is desired.
