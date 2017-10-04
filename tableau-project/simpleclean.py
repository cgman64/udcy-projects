# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 21:31:43 2017

@author: Gabriel
"""

import csv
import os
import pandas as pd

flight = "C:/Users/Gabriel/workspace/udacity-projects/tableau-project/4241_airline_delay_causes.csv"
flight = pd.read_csv(flight)

# Write function that tells me which columns have NA values.
ctr = 0
for elem in flight.columns.values:
    if flight[elem].isnull().values.any():
        print "column \"" + elem + "\" has null values."
        ctr += 1
print "\n" + str(ctr) + " out of " \
+ str(len(flight.columns.values)) + " columns have null values."

flight = flight.iloc[:,:-1]
#Number of rows:
#Before removing null values
print(str(len(flight)) + ' rows before removing null values.')
#After
print(str(len(flight.dropna())) + ' rows after removing null values.')
print('Difference in number of rows: ' + str(len(flight) - len(flight.dropna())))


flight = flight.dropna()

PATH = "C:/Users/Gabriel/workspace/udacity-projects/tableau-project/airline_delay_causes.csv"
flight.to_csv(PATH)