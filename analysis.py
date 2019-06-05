#!/usr/bin/env python3

# May first need:
# In your VM: sudo apt-get install libgeos-dev (brew install on Mac)
# pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon



def plot_map(m, info):
    map_cmap = info['cmap']
    vmin = info['vmin']
    vmax = info['vmax']
    data = info['data']
    colors = {}
    statenames = []
    for shapedict in m.states_info:
      statename = shapedict['NAME']
      # skip DC and Puerto Rico.
      if statename not in ['District of Columbia', 'Puerto Rico']:
          num = data[statename]
          colors[statename] = map_cmap(( num - vmin )/( vmax - vmin))[:3]
      statenames.append(statename)

    ax = plt.gca() # get current axes instance

    for nshape, seg in enumerate(m.states):
        # skip Puerto Rico and DC
        if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)
    ax.legend().set_visible(False)
    plt.title(info['title'])
    plt.savefig(info['name'])


def plot_part1():
  """
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

  ts = pd.read_csv("time_data.csv")
  # Remove erroneous row.
  print(ts)
  plt.figure(figsize=(12,5))
  ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
  ts.set_index(['date'],inplace=True)

  ax = ts.plot(title="President Trump Sentiment on /r/politics Over Time",
          color=['green', 'red'],
         ylim=(0, 1.05))
  ax.plot()

  plt.savefig("part1.png")


plot_part1()



"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

state_data = pd.read_csv("state_data.csv")

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"
The rename the files to get rid of the ?raw=true
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

pos_info ={
"title": "Positive Trump Sentiment across the US",
"data": pos_data,
"name": "pos_map.png",
"vmin": .3,
"vmax": .37,
"cmap": plt.cm.Greens
}
neg_info = {
"title": "Negative Trump Sentiment across the US",
"data": neg_data,
"name": "neg_map.png",
"vmin": .87,
"vmax": .95,
"cmap": plt.cm.Reds
}

diff_data = {}
for state in pos_data:
  diff_data[state] = abs(pos_data[state] - neg_data[state])

diff_info = {
"title": "Difference in Trump Sentiment across the US",
"data": diff_data,
"name": "diff_map.png",
"vmin": .56,
"vmax": .63,
"cmap": plt.cm.Blues
}
plot_map(m, pos_info)
plot_map(m, neg_info)
plot_map(m, diff_info)



# # SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# # (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

# """
# PART 4 SHOULD BE DONE IN SPARK
# """

"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

story = pd.read_csv("submission_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5a.png")

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

story = pd.read_csv("comment_score.csv")
plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

# ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
# ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
# plt.legend(loc='lower right');

# plt.xlabel('President Trump Sentiment by Comment Score')
# plt.ylabel("Percent Sentiment")
# plt.savefig("plot5b.png")