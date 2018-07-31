from sqlalchemy import create_engine, MetaData, Table, Column, inspect, select
from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy import Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from pprint import pprint
from sqlalchemy import or_

def find_syndrome(number):
    #Find the syndrome and synonyms of the disorder from the id_orpha_disorder number
    #Number should be a string
    #Returns the found results in a list of strings

    #Create an engine to open the data base
    engine = create_engine('sqlite:///minerva2_info.db', echo=False)
    conn = engine.connect()
    Base = declarative_base(engine)

    #get table from metadata
    metadata = MetaData(engine, reflect = True)
    #Load in the two relevant tables
    dtable = metadata.tables['orpha_disorders']
    stable = metadata.tables['orpha_disorder_synonyms']

    #Create somewhere to store the results
    results = []        #Dont know how many results we are going to get

    #Find the name of the disorder
    #Query the table to find where the number is equal to the query value
    select_st = select([dtable]).where(dtable.c.id_orpha_disorder == number)
    res = conn.execute(select_st)
    for _row in res:
        results.append(_row[1])     #The syndrome name is in index one

    #Find synonyms of the disorder
    #Query the table to find where the number is equal to the query value
    select_st = select([stable]).where(stable.c.id_orpha_disorder == number)
    res = conn.execute(select_st)
    for _row in res:
        results.append(_row[2])     #The syndrome name in in index two

    return results
