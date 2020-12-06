import sqlite3
import os
import sys
import yaml
import argparse


def readPTXAnalysis(filepath):
    kernel_map = {}
    doc = yaml.load(open(filepath, 'r'))
    for func in doc:
        blocks = []
        try:
            for block in func['Blocks']:
                inst_map = {}
                instructions = list(block.values())[0]
                for inst in instructions:
                    if inst_map.get(inst) == None:
                        inst_map[inst] = 1
                    else:
                        inst_map[inst] += 1
                blocks.append(inst_map)
        except:
            continue
        kernel_map[func['Name']] = blocks
    return kernel_map

def readBlockExecutions(path, kernel_map):
    kernel_list = []
    for line in open(path).readlines():
        if line.startswith('#'):
            continue
        vec = line.split(',')
        name = vec[0]
        gridX = int(vec[1])
        gridY = int(vec[2])
        gridZ = int(vec[3])
        blockX = int(vec[4])
        blockY = int(vec[5])
        blockZ = int(vec[6])
        sharedMemory = int(vec[7])
        time = int(vec[8])
        profiling_mode = int(vec[9])
        
        counters = vec[10:]
        blockIdx = 0
        value_map = { 'gX' : gridX, 'gY' : gridY, 'gZ' : gridZ, 
                     'bX' : blockX, 'bY' : blockY, 'bZ' : blockZ, 'shm' : sharedMemory, 'time' : time}
        blocks = kernel_map[name]
        for count in counters:
            count = int(count)
            block = blocks[blockIdx]
            for inst in block:
                if value_map.get(inst) == None:
                    value_map[inst] = block[inst] * count
                else:
                    value_map[inst] += block[inst] * count
            blockIdx += 1
        kernel_list.append((name, value_map))

    return kernel_list

def readCUDAFluxMetrics(path):
        try:
            kernel_map = readPTXAnalysis(os.path.join(path,'PTX_Analysis.yml'))
        except FileNotFoundError:
            print("\033[1;31mCould not read PTX_Analysis.yml\033[0m")
            return None
        except yaml.scanner.ScannerError:
            print("\033[1;31mMalformed PTX_Analysis.yml\033[0m")
            return None
        try:
            kernel_list = readBlockExecutions(os.path.join(path, 'bbc.txt'), kernel_map)
        except:
            print("\033[1;31mError!\033[0m")
            raise
            return None

        return kernel_list


def processArgs():
    # Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', metavar='<path to logdir>', default='log')
    parser.add_argument('-tag', metavar='<tag>', required=True)
    parser.add_argument('-o', metavar='<output-db-name>', default='CUDAFluxMetrix.db')
    # Parse
    return parser.parse_args()


def main():
    args = processArgs()
    print(args)
    dbPath = args.logdir + '/log.db'
    tag = args.tag

    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()

    conn_mem = sqlite3.connect(args.o)
    cur_mem = conn_mem.cursor()
    cur_mem.execute('''create table fluxmetrics
                    (bench text not null,
                    app text not null,
                    dataset text not null,
                    lseq integer not null,
                    name text not null,
                    metric text not null,
                    value double not null)''')

    cur.execute("select bench,app,dataset from benchlog where tag=?",(tag,))

    res = cur.fetchall()
    key_set = set(res)

    for bench,app,data in key_set:
        print("Processing", bench,app,data)
        tmp = cur.execute("select logdir from benchlog where bench=? and app=? and dataset=? and tag=?", (bench,app,data,tag))
        res = cur.fetchall()
        for item in res:
            print (item)
            kernel_list = readCUDAFluxMetrics(args.logdir + '/../' + item[0])
            if kernel_list is None:
                continue
            lseq = 0
            for item in kernel_list:
                for metric in item[1]:
                    value = float(item[1][metric])
                    cur_mem.execute("insert into fluxmetrics values(?,?,?,?,?,?,?)", (bench, app, data, lseq, item[0], metric, value))
                lseq += 1

    conn_mem.commit()

    #tmp = cur_mem.execute("select * from fluxmetrics")
    #res = cur_mem.fetchall()
    #for item in res:
    #    print(item)
    print("Done")


if __name__ == '__main__':
    main()
