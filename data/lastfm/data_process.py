import numpy as np
import pandas as pd
import sys


userID_list = []
artistID_list = []
tagID_list = []
timestamp_list = []
with open("original_files/user_taggedartists-timestamps.dat", 'r') as file:
    for line in file:
        line = line.strip("\n")
        userID, artistID, tagID, timestamp = line.split("\t")
        userID_list.append(userID)
        artistID_list.append(artistID)
        tagID_list.append(tagID)
        timestamp_list.append(timestamp)

df = pd.DataFrame({'userID': userID_list[1:], 'artistID': artistID_list[1:], 'tagID': tagID_list[1:], 'timestamp': timestamp_list[1:]})
uni_user_list = df['userID'].unique().tolist()
uni_item_list = df['artistID'].unique().tolist()
print(len(uni_item_list))
print(df[df['artistID']=='52'])
sys.exit()
temp = df[(df['userID']=="2") & (df['artistID']=='52')]
print(temp)