# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:52:21 2019

@author: MUKHESH
"""
import json
import sqlite3
import mysql.connector as mysql
from datetime import datetime

data_time_frame='2015-01'
transaction_sql=[]
client=sqlite3.connect('{}.db'.format(data_time_frame))

#client=mysql.connect(host='localhost',user='root',password='root')
cursor=client.cursor()
#cursor.execute('''CREATE DATABASE IF NOT EXISTS {}'''.format('RC_'+data_time_frame.split('-')[0]))
#cursor.execute('''USE {}'''.format('RC_'+data_time_frame.split('-')[0]))

def create_table():
    cursor.execute('''CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT,comment_id TEXT UNIQUE,parent TEXT,comment TEXT,subreddit TEXT,unix_time INT,score INT)''')
    #cursor.execute('''CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY,comment_id TEXT UNIQUE,parent TEXT,comment TEXT,subreddit TEXT,unix_time INT ,score INT)''')
    
def cleaning_body(body):
    text=body.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    return text

def filter_ids(pid):
    try:
        sql='''SELECT COMMENT FROM PARENT_REPLY WHERE COMMENT_ID='{}' LIMIT 1'''.format(pid)
        cursor_object=cursor.execute(sql)
        fetched=cursor_object.fetchone()
        if fetched != None:
            return fetched[0]
        else:
            return False
    except Exception as e:
        return False

def acceptable_data(data):
    if len(data.split(' '))<50 or len(data)<1:   
        return False
    elif data=='[deleted]' or data=='[removed]':
        return False
    else:
        return True

def existing_scores(pid):
    try:
        sql='''SELECT SCORE FROM PARENT_REPLY WHERE PARENT_ID='{}' LIMIT 1'''.format(pid)
        cursor_object=cursor.execute(sql)
        fetched=cursor_object.fetchone()
        if fetched != None:
            return fetched[0]
        else: return False
    except Exception as e:
        return False


def transaction_blr(sql):
    try:
        global transaction_sql
        transaction_sql.append(sql)
        if len(transaction_sql)>1000:
            cursor.execute('''BEGIN TRANSACTION''')
            for sql in transaction_sql:
                try:
                    cursor.execute(sql)
                except Exception as e:
                    print(str(e))
                    pass
            client.commit()
            transaction_sql=[]
    except Exception as e:
        print(str(e))

def sql_insert_replace_data(comment_id,parent_id,parent_text,clean_text,utc,score,subreddit):
    try:
        sql='''UPDATE parent_reply SET parent_id="{}",comment_id="{}",parent="{}",comment="{}",subreddit="{}",unix_time={},score={} where parent_id="{}"'''.format(parent_id,comment_id,parent_text,clean_text,subreddit,utc,score,parent_id)
        transaction_blr(sql)
    except Exception as e:
        pass

def sql_insert_data(comment_id,parent_id,parent_text,clean_text,utc,score,subreddit):
    try:
        sql='''INSERT INTO parent_reply (parent_id,comment_id,comment,subreddit,unix_time,score) VALUES("{}","{}","{}","{}",{},{})'''.format(parent_id,comment_id,clean_text,subreddit,int(utc),score)
        transaction_blr(sql)
    except Exception as e:
        pass
    
def sql_insert_parent_data(comment_id,parent_id,parent_text,clean_text,utc,score,subreddit):
    try:
        sql='''INSERT INTO parent_reply (parent_id,comment_id,parent,comment,subreddit,unix_time,score) VALUES("{}","{}","{}","{}","{}",{},{})'''.format(parent_id,comment_id,parent_text,clean_text,subreddit,int(utc),score)
        transaction_blr(sql)
    except Exception as e:
        pass    

row_count=0   
paired_row=0
if __name__=='__main__':
    create_table()
    with open('C:/Users/MUKHESH/OneDrive/Documents/Python Scripts/Reddit_Comments/RC_{}/RC_{}'.format(data_time_frame.split('-')[0],data_time_frame),buffering=10000) as f:
        for row in f:
            #print(row)
            data=json.loads(row)
            row_count=row_count+1
            parent_id=data['parent_id'].split('_')[1]
            body=data['body']
            score=data['score']
            subreddit_id=data['subreddit']
            comment_id=data['name']
            utc=data['created_utc']
            clean_text=cleaning_body(body)
            parent_text=filter_ids(parent_id)
            
            existing_score=existing_scores(parent_id)
            if existing_score:
                if score>existing_score:
                    if acceptable_data(clean_text):
                        sql_insert_replace_data(comment_id,parent_id,parent_text,clean_text,utc,score,subreddit_id)
            else:
                if acceptable_data(clean_text):
                    if parent_text:
                        if score>=1:
                            sql_insert_parent_data(comment_id,parent_id,parent_text,clean_text,utc,score,subreddit_id)
                            paired_row+=1
                            
                    else:
                        sql_insert_data(comment_id,parent_id,parent_text,clean_text,utc,score,subreddit_id)
            if row_count%1000==0:            
                print('no. of rows:'+str(row_count)+' no. of paid rows:'+str(paired_row)+' the time is:'+str(datetime.now()))