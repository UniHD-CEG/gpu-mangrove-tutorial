#!/usr/bin/env python3
#In this preprocessing script, it was  n range (0, len(new_time_list)-1):
#is that when you are considering all the duplicates then the there are a lot of small contributions from each power(i)*time_interval(i) value which in the end leads to#slighlty different resolution.
#Meaning a sum of a lot of decimals. Of course, when this is divided by a big number the resolution is also changed. So, basically, I tried in any step to round the val#ues using three digits.

import pandas as pd
import numpy as np
import os
import yaml
from datetime import timedelta
import itertools
import sqlite3
import argparse
import re

def processArgs():
    # Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', metavar='<path to benchlog db>', default='log')
    parser.add_argument('-tag', metavar='<tag>', required=True)
    # parser.add_argument('-o', metavar='<output-db-name>', default='kernelTime.db')
    # Parse
    return parser.parse_args()


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)
 
    return None


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print(e)


def getBenchList(db_path, tag, benchmark):
    #Give the benchmark path
    print(db_path, tag, benchmark)
    log = pd.read_csv(db_path)
    #Correct the tag you gave
    log_EP = log[log['Tag'] == tag]
    return log_EP


def prepare_file(filename: str):
    output = ''
    with open(filename) as f:             
        for line in f:
            if line.rfind(")") != -1:
                if line.rfind("]") != -1:
                    last_position = line.rfind("]")
                    n = re.search(",", line)
                    first_position = n.start()
                    name = line[first_position+1:last_position]
                    new_name = name.replace(',', '')
                    line = line.replace(name,new_name)
                    output = output + (line)
                else:
                    last_position = line.rfind(")")
                    n = re.search(",", line)
                    first_position = n.start()
                    name = line[first_position+1:last_position]
                    new_name = name.replace(',', '')
                    line = line.replace(name,new_name)
                    output = output + (line)
            else:
                output = output + (line)

    file = open(filename, 'w', encoding='utf-8')
    file.write(output)
    file.close()



def read_energy():
    final_results = []
    columns = ["file","kernel","launch","measurement","t","p"]
    #Create dataframe
    DATA = pd.read_csv("energy.csv", delimiter=',', header=None, names = columns)
    #Delete all the duplicated headers
    DATA = DATA[DATA.file != 'kernel_file']
    #Delete all the duplicated power values
    DATA = DATA.drop_duplicates(subset = ["file","kernel","launch","p"], keep='last') 
    #Group into different dataframes by checking whether the measurement value is increasing or it is zero.
    df_list = [x[1] for x in DATA.groupby((DATA.measurement.astype(int).diff() <= 0).cumsum())]
    for idx in range(len(df_list)):
        #print(df_list[idx])
        Grouped_Data = df_list[idx].groupby(DATA.launch.astype(int))
        #Loop over all the existed groups, basically it is going over all launches
        for i, group in Grouped_Data:
            #Check whether there are more than one value. If there is only one value, then there are not power measurements(SHORT KERNEL)
            if Grouped_Data.get_group(i).measurement.count() == 1:
                #Name of file, use the element of list. There is only one element
                filename = list(Grouped_Data.get_group(i).file)[0]
                #Name of kernel, use the element of list. There is only one element
                name = list(Grouped_Data.get_group(i).kernel)[0]
                #Launch sequence, use the element of list. There is only one element
                launch_seq = list(Grouped_Data.get_group(i).launch)[0]
                #Since, there are no power values, then ,energy_average, power_peak, power_average and samples are zero
                power_peak = 0
                power_weighted_mean = 0
                samples = 0
                elapsed_time = 0
                hum_elapsed_time = str(timedelta(microseconds=elapsed_time))
                #Save all the data in results variable
                results = [filename, name, launch_seq, power_peak, power_weighted_mean, samples, elapsed_time, hum_elapsed_time]   
                final_results.append(results)
            else:
                #Name of file, use the element of list. There is only one element
                filename = list(Grouped_Data.get_group(i).file)[0]
                #Name of kernel, the first element of the list is used, since the other is the same
                name = list(Grouped_Data.get_group(i).iloc[0:1].kernel)[0]
                #Launch sequence, the first element of the list is used, since the other is the same
                launch_seq = list(Grouped_Data.get_group(i).iloc[0:1].launch)[0]
                #Power_list, select all the power values. 
                power_list = list(Grouped_Data.get_group(i).p)
                #Time_list, select all the power values. 
                time_list = list(Grouped_Data.get_group(i).t)
                #Create a new list where the values finally will be saved
                new_power_list = []
                new_time_list = []
                #Loop over all the power values in the list and create a new list with float numbers. I keep as many decimal as the power measurement gives me (3 decimals)
                for item1 in power_list:
                    new_power_list.append(float(item1))
                #Loop over all the power values in the list and create a new list with integer numbers
                for item2 in time_list:
                    new_time_list.append(int(item2))
                #Create time_interval list where every time interval of each measurement will be saved. IMPORTANT for correct energy calculation
                time_intervals = []
                #Take always the first element without subtraction.!Basically, this needs in order to be subctracted. It is simpply zero.
                time_intervals.append(new_time_list[0])
                #Loop over the new_time_list and subtract the timings, to calculate the time interval of each measurement, PLEASE use other index (k) than the existed ones(otherwise ERROR)
                #Moreover, I used the first element to be zero. So, it is a - 0 = a, the first time interval
                for k in range (0, len(new_time_list)-1):
                    time_intervals.append(new_time_list[k+1] - new_time_list[k])
                #Multiply each power measurement by its time interval. I also round the product of power times intervals using simple three decimals
                #The first element of both time interval and power list is zero, so its product is also zero. Therefore, it does not contribute to the sum
                energy_list = [round(new_power_list[i]*time_intervals[i],3) for i in range(len(new_power_list))]
                #Find the maximum power value from the float list of power values. Search between all values apart from the first one
                power_peak = max(new_power_list)
                #Calculate the average power of new_list using Average function. Round it and keep only three digits since the initial value has only three degits
                #The following is calculating the power weighted mean
                power_weighted_mean = round(round(sum(energy_list),3)/sum(time_intervals),3)
                #The number of samples is the last value of the measurements as given
                samples = Grouped_Data.get_group(i).iloc[-1].measurement
                elapsed_time = int(Grouped_Data.get_group(i).iloc[-1].t)
                hum_elapsed_time = str(timedelta(microseconds=elapsed_time))
                results = [filename, name, launch_seq, power_peak, power_weighted_mean, samples, elapsed_time, hum_elapsed_time]     
                final_results.append(results)
    return final_results


def read_energy_counter():
    final_results = []
    columns = ["file","kernel","launch"]
    #Create dataframe
    DATA = pd.read_csv("energy_counter.csv", delimiter=',', header=None, names = columns)
    #Delete all the duplicated headers
    DATA = DATA[DATA.file != 'kernel_file']
    for i in range(len(DATA)):
        filename = DATA['file'].values
        name = DATA['kernel'].values
        launch_counter = DATA['launch'].values
        results = [filename, name, launch_counter]
    final_results.append(results)
    return final_results


# functions for filtering
def removeKernelwithFewPowerMeasurements(dbPath):
    connection = sqlite3.connect(dbPath)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE power_filtered AS SELECT * from power_complete where NOT samples < 1700 AND samples > 0;")
    results = cursor.fetchall()
    for r in results:
        print(r)
    cursor.close()
    connection.close()


def checkKernelPowerMeasurements(dbPath):
    with sqlite3.connect(dbPath) as conn:
    #Check if there are not correct values, meaning zeros or very short measurements (not included)
    #It should not print anything in order to be correct
        cur = conn.cursor()
        cur.execute("SELECT * from power_filtered where samples < 1700 AND samples > 0;")
        for res in cur.fetchall():
            print(res)


def checkNotMatchingSamples(kernelPowerDB, featureDB):
    with sqlite3.connect(kernelPowerDB) as conn:
        cur = conn.cursor()
        cur.execute("attach ? as ft;", (featureDB,))
        cur.execute("""
select count(1) from 
 power_filtered as p left outer join ft.fluxfeatures as f 
     on p.bench = f.bench and p.app = f.app and p.dataset = f.dataset and p.lseq = f.lseq 
where f.bench is null;
        """)
        print('Missing Features:')
        for res in cur.fetchall():
            print(res)
        print('Missing Power:')
        cur.execute("""
select count(1) from 
 ft.fluxfeatures as f left outer join power_filtered as p 
     on p.bench = f.bench and p.app = f.app and p.dataset = f.dataset and p.lseq = f.lseq 
where p.bench is null;
        """)
        for res in cur.fetchall():
            print(res)


# actual processing
if __name__ == '__main__':
    benchmarks = [ 'polybench-gpu-1.0']
    root_dir = os.getcwd()
    dataset = []
    dataset_counter = []

    args = processArgs()
    db_path = args.logdir + '/log.db'
    tag = args.tag

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("select bench,app,dataset,build from benchlog where tag=?", (tag,))

    res = cur.fetchall()
    key_set = set(res)
    # print(key_set)

    for bench,app,data,build in key_set:
        os.chdir(root_dir)
        tmp = cur.execute("select logdir from benchlog where bench=? and app=? and dataset=? and build=? and tag=?", (bench,app,data,build,tag))
        res = cur.fetchall()
        for item in res:
            # print(item, item[0])
            os.chdir(root_dir)
            # print(os.getcwd())
            path = os.path.join(root_dir, item[0])
            try:
                os.chdir(path)
            except:
                print("\033[1;31mCould not change to " + item[0] + "\003[0m")
            try:
                # print(os.getcwd())
                prepare_file("energy.csv")
                prepare_file("energy_counter.csv")
                kernel_list = read_energy()
                counter_list = read_energy_counter()
            except Exception as e:
                print("\033[1;31mCould not read energy.csv\033[0m")
                print(str(e))
                continue
            for row_energy in kernel_list:
                row_energy.insert(0, data)
                row_energy.insert(0, app)
                row_energy.insert(0, bench)
                # print(row_energy)
                dataset.append(row_energy)
            for row_energy in counter_list:
                row_energy.insert(0, data)
                row_energy.insert(0, app)
                row_energy.insert(0, bench)
                dataset_counter.append(row_energy)
    print("Finished")
    os.chdir(root_dir)

    # create and fill power database
    labels = ["bench","app","dataset","filename","kernelname","lseq","max_power","aver_power","samples","elapsed_time","hum_elapsed_time"]
    df = pd.DataFrame(dataset, columns = labels)

    #print (df[:10])
    database = "preprocessed_power.db"
    sql_create_power_table =  """ CREATE TABLE IF NOT EXISTS power ( 
                                     bench TEXT, 
                                     app TEXT, 
                                     dataset TEXT, 
                                     filename TEXT, 
                                     kernelname TEXT,
                                     lseq INTEGER, 
                                     max_power REAL, 
                                     aver_power REAL, 
                                     samples INTEGER, 
                                     elapsed_time INTEGER, 
                                     hum_elapsed_time TEXT 
                              ); """


    conn = create_connection(database)
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_power_table)
    else:
        print("Error! cannot create the database connection.")

    df.to_sql('power', conn, if_exists='append', index = False)  


    # create and fill power_counter database
    labels = ["bench","app","dataset","filename","kernelname","lcounter"]
    df = pd.DataFrame(dataset_counter, columns = labels)

    # print (df)
    benchs = []
    apps = []
    datasets = []
    filenames = []
    kernelnames = []
    lcounters = []
    final_row = []
    for index, row in df.iterrows():
        bench = row.bench
        app = row.app
        dataset = row.dataset
        for filename in row.filename:
            benchs.append(bench)
            apps.append(app)
            datasets.append(dataset)
            filenames.append(filename)
        for kernelname in row.kernelname:
            kernelnames.append(kernelname)
        for lcounter in row.lcounter:
            lcounters.append(lcounter)


    df_new = pd.DataFrame({"bench" : benchs, "app" : apps, "dataset" : datasets, "filename" : filenames,"kernelname" : kernelnames, "lcounter" : lcounters}, columns = labels)

    #print(df_new)

    database = "preprocessed_power_counter.db"

    sql_create_power_table_counter =  """ CREATE TABLE IF NOT EXISTS power_counter ( 
bench TEXT, 
app TEXT, 
dataset TEXT, 
filename TEXT, 
kernelname TEXT,
lcounter INTEGER 
); """


    conn = create_connection(database)
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_power_table_counter)
    else:
        print("Error! cannot create the database connection.")

    df_new.to_sql('power_counter', conn, if_exists='append', index = False)


    ## Merge databases
    # Get connections to the databases
    db_a = sqlite3.connect('preprocessed_power.db')
    db_b = sqlite3.connect('preprocessed_power_counter.db')


    df_1 = pd.read_sql_query(con=db_b, sql = """SELECT lcounter FROM power_counter""")
    df_2 = pd.read_sql_query(con=db_a, sql = """SELECT * FROM power""")

    result = pd.concat([df_2, df_1], axis=1)  #, join_axes=[df_2.index])

    result['actual_time'] = result['elapsed_time']/result['lcounter']
    result.to_sql('power_complete', con=db_a, if_exists='replace')

    print(result[:10])

    ## Filter results
    kpDB = 'preprocessed_power.db'
    removeKernelwithFewPowerMeasurements(kpDB)
    print('Samples with too few power measurements:')
    checkKernelPowerMeasurements(kpDB)
