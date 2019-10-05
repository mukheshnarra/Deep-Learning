# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:29:23 2019

@author: MUKHESH
"""

import sqlite3
import pandas as pd

timeframe='2015-01'
limit=15000
cur_length=limit
counter=0
unix=0
test_done=False

connection=sqlite3.connect('{}.db'.format(timeframe))
c=connection.cursor()

while cur_length==limit:
    data=pd.read_sql('SELECT * FROM parent_reply WHERE unix>{} AND parent IS NOT NULL AND score>1 ORDER BY unix ASC LIMIT {}'.format(unix,limit),connection)
    unix=data.tail(1)['unix'].values[0]
    limit=len(data)
    cur_length=limit
    if not test_done:
        
        with open('test.from','a',encoding='utf-8') as f:
            for content in data['parent'].values:
                f.write(content+'\n')
        
        with open('test.to','a',encoding='utf-8') as f:
            for content in data['comment'].values:
                f.write(content+'\n')
        test_done=True
    
    else:
        with open('train.from','a',encoding='utf-8') as f:
            for content in data['parent'].values:
                f.write(content+'\n')
        
        with open('train.to','a',encoding='utf-8') as f:
            for content in data['comment'].values:
                f.write(content+'\n')
                
    counter+=1
    if counter%20==0:
        print(str(counter*limit)+' Finished rows processing')
        





