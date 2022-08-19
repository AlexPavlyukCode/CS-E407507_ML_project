import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('tweet_data.csv', encoding='latin-1')

# Filtering the dataframe
df.drop(['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
         '_last_judgment_at', 'gender', 'gender:confidence', 'profile_yn',
         'profile_yn:confidence', 'created', 'description', 'fav_number',
         'gender_gold', 'link_color', 'name', 'profile_yn_gold', 'profileimage',
         'retweet_count', 'sidebar_color', 'tweet_coord', 'tweet_count',
         'tweet_created', 'tweet_id', 'tweet_location'], axis=1, inplace=True)
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={"user_timezone": "timezone"}, inplace=True)

# At this point we have the following dataframe:
#   1. 12,252 rows
#   2. 2 columns: text and timezone
#   3. 156 unique timezones

# Encoding timezones with value from 0 to 155
le = LabelEncoder()
df["timezone"] = le.fit_transform(df["timezone"])


tweets = df["text"].to_numpy()
timezones = df["timezone"].to_numpy
