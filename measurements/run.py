#!/usr/bin/env python3

import sys
import os
import argparse
import shutil
import time
import datetime
import socket
import hashlib
import sqlite3

logFolder = 'log'
logDB = 'log.db'
dbPath = os.path.join(logFolder, logDB)
logTable = 'benchlog'

binaryFolder = 'bin'
tmpFolder = 'tmp'
appArgumentsFile = 'app_arguments.txt'

def main():
    args = processArgs()
    ret_val = -1

    print ('Benchmark:', args.suite)
    print ('Application:', args.app)
    print ('Binary build:', args.build)
    print ('Dataset:', args.dataset)
    print ('Iterations:', args.iter)
    print ('Runner:', args.runner)
    print ('Tag:', args.tag)

    # Get hostname
    hostname = socket.gethostname().split('.')[0]

    # Build path of binary
    binPath = os.path.join('bin', args.build, args.app)

    # Tee cmd
    teeCmd = ' 2>&1 | tee ' + 'output.txt'

    #l Read runfile and execute
    with open(os.path.join(args.suite, appArgumentsFile), 'r') as appArgFile:
        for line in appArgFile:
            app, dataset, arguments = readCommand(line)
            if(app != args.app):
                continue
            if(dataset != args.dataset):
                continue

            # Open Sqlite Database
            conn = sqlite3.connect(dbPath)
            initTable(conn)

            cur = conn.cursor()

            for i in range(0, args.iter):
                print('Iteration', i)

                # Create tmp folder
                tmpPath = os.path.join(args.suite, tmpFolder)
                if os.path.exists(tmpPath):
                    shutil.rmtree(tmpPath)
                os.makedirs(tmpPath)

                timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
                hashPrefix = hashlib.sha1(timestamp.encode()).hexdigest()
                hashPrefix = hashPrefix[0:4]
                logPath = os.path.join(logFolder, hashPrefix, timestamp + '_' + hostname + '_' + app)
                cmd = args.runner + ' ' + os.path.join('..', binPath) + ' ' + arguments + ' ' + teeCmd
                print (cmd)
                sys.stdout.flush()
                os.chdir(tmpPath)
                start = datetime.datetime.now()
                ret_val = os.system(cmd)
                stop = datetime.datetime.now()
                os.chdir('../..')


                values = (args.suite, app, dataset, args.build,
                         args.runner, hostname, args.tag, 
                         start, stop, logPath)
                cur.execute("insert into benchlog values(?,?,?,?,?,?,?,?,?,?)", values)
                conn.commit()

                # Move all tmpfiles to logs
                if not os.path.exists(logFolder):
                    os.makedirs(logFolder)

                os.makedirs(logPath)
                for file in os.listdir(tmpPath):
                    os.rename( os.path.join(tmpPath, file), os.path.join(logPath, file))

                # Remove tmp folder
                os.rmdir(tmpPath)
            conn.close()

    # Finish
    sys.exit(ret_val)

def processArgs():
    # Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--suite', type=str, required=True, 
                        help='benchmarksuite of application to be executed')
    parser.add_argument('-a', '--app', metavar='<application>', type=str, required=True, 
                        help='application to be executed')
    parser.add_argument('-b', '--build', metavar='<build>', type=str, required=True, 
                        help='determines the build that will be executed')
    parser.add_argument('-d', '--dataset', metavar='<dataset>', type=str, required=True, 
                        help='determines the for the executable')
    parser.add_argument('-t', '--tag', metavar='<tag>', type=str, default='', 
                        help='tag will be stored in db, useful for later processing')
    parser.add_argument('-i', '--iter', metavar='<iterations>',type=int, default=1,
                        help='execute the benchmark <iterations> times')
    parser.add_argument('-r', '--runner', metavar='<runner>',type=str, default='',
                        help='wrapper for benchmark')
    # Parse
    return parser.parse_args()


def initTable(conn, table = logTable):
    cur = conn.cursor()

    cur.execute('create table if not exists ' + logTable + ''' 
        (bench text not null,
        app text not null,
        dataset next not null,
        build text not null,
        runner text,
        hostname text not null,
        tag text,
        start date not null,
        stop date not null,
        logdir text not null);''')
    conn.commit()
    print("DB: init done")

def readCommand(line):
    tmp = line.split()
    app = tmp[0]
    dataset = tmp[1]
    argument = ' '.join(tmp[2:])
    return app, dataset, argument

if __name__ == '__main__':
    main()

