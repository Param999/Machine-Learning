#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:53:30 2017

@author: param
"""

import pandas as pd
import quandl as q
import math

df = q.get('WIKI/GOOGL')
#print(df.tail())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#print(df.tail())

df['HL_PERCENT'] = (df['Adj. High'] - df['Adj. Low']) /df['Adj. Low'] *100
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open'] *100

df = df[['Adj. Close','HL_PERCENT','Percent_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

print(len(df))
forecast_out = int(math.ceil(0.01*len(df))) #10% of total data
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace = True)
print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values