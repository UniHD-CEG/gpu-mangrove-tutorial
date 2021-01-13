import sqlite3
import os
import sys
import argparse


def processArgs():
    # Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', metavar='<path to benchlog db>', default='log')
    parser.add_argument('-tag', metavar='<tag>', required=True)
    parser.add_argument('-o', metavar='<output-db-name>', default='kernelPower.db')
    # Parse
    return parser.parse_args()


def main():
    args = processArgs()
    dbPath = args.logdir + '/log.db'
    tag = args.tag

    print ("using tag ", tag)

    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()

    conn_mem = sqlite3.connect(args.o)
    cur_mem = conn_mem.cursor()
    cur_mem.execute('''create table kernelpower
                    (bench text not null,
                    app text not null,
                    dataset text not null,
                    build text not null,
                    lseq integer not null,
                    power integer not null)''')

    cur_mem.execute("PRAGMA synchronous = OFF")
    #cur_mem.execute("PRAGMA journal_mode = MEMORY")

    cur.execute("select bench,app,dataset,build from benchlog where tag=?", (tag,))

    res = cur.fetchall()
    key_set = set(res)

    for bench,app,data,build in key_set:
        tmp = cur.execute("select logdir from benchlog where bench=? and app=? and dataset=? and build=? and tag=?",
            (bench,app,data,build,tag))
        res = cur.fetchall()
        for item in res:
            print(item)
            try:
                for line in open(os.path.join(args.logdir,'../',item[0],'kernelpower.csv'), 'r'):
                    line = line.split(',')
                    try:
                        launch_seq = int(line[0])
                        power = float(line[1])
                    except IndexError:
                        print("\033[1;31mIndexError!\033[0m")
                        continue
                    except ValueError:
                        print("\033[1;31mValueError!\033[0m")
                        continue
                    tmp = cur_mem.execute("insert into kernelpower values(?,?,?,?,?,?)", (bench, app, data,build, launch_seq, power,)) 
            except IOError:
                print("\033[1;31mIOError!\033[0m")
                continue
            conn_mem.commit()

    tmp = cur_mem.execute("select  bench,app,dataset,build,lseq,avg(power) from kernelpower group by bench,app,dataset,build,lseq")
    res = cur_mem.fetchall()
    for item in res:
        print(item)


if __name__ == '__main__':
    main()
