#!/usr/bin/env python3

import os
import sys
import time
import signal
from argparse import ArgumentParser
import yaml

global interrupt
interrupt = False

def main():
    global interrupt

    args = processArgs()

    # If allocation command is given, allocate node(s) in batch system 
    if(args.alloc_cmd != ""):
        print(args.alloc_cmd)

    if args.resume == '':
        args.resume = None

    doc = yaml.load(open(args.list))

    for suite in doc:
        applist = doc[suite]
        for app in doc[suite]:
            for build in args.build.split(','):
                # TODO: check if binary exists
                for data in doc[suite][app]:
                    iter = args.iter
                    tag = args.tag

                    # In case a list is resumed at a specific point
                    if args.resume != None:
                        r_runner, r_suite, r_app, r_build, r_data, r_i = args.resume.split(',')
                        if r_runner == runner and suite == r_suite and \
                                app == r_app and r_build == build and \
                                data == r_data:
                            args.resume = None
                        else:
                            continue

                    # In case of Interrupt let finish after last command is done
                    if interrupt == True:
                        print( "Resume at:", args.runner + ',' + suite + ',' + \
                              app + ',' + build + ',' + data)
                        # If free command is given release the allocation
                        if(args.free_cmd):
                            print(args.free_cmd)
                        return

                    driver_cmd = ' '.join([args.queue_cmd, "run.py", '-s', suite, '-a', app, '-b', build, \
                                           '-d', data, '-t', tag, '-i', str(iter)])
                    if args.runner != '':
                        driver_cmd += " -r '" + args.runner + "'"
                    print(driver_cmd)
                    #os.system(driver_cmd)

    # If free command is given release the allocation
    if(args.free_cmd):
        print(args.free_cmd)

    return

def processArgs():
    # Setup Arguments
    parser = ArgumentParser()
    #group = parser.add_mutually_exclusive_group()
    parser.add_argument('-l', '--list', metavar='<benchmarks.yaml>', type=str)
    parser.add_argument('-s', '--session')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('-q', '--queue-cmd', type=str, default='')
    parser.add_argument('-a', '--alloc-cmd', type=str, default='')
    parser.add_argument('-f', '--free-cmd', type=str, default='')
    parser.add_argument('-b', '--build', metavar='<build list>', type=str, required=True)
    parser.add_argument('-r', '--runner', metavar='<runner>', type=str, default='')
    parser.add_argument('-t', '--tag', type=str, default='changeme')
    parser.add_argument('-i', '--iter', type=int, default=1)
    return parser.parse_args()

# Handle CTRL-C
def signal_handler(sig, frame):
    global interrupt
    print("\nInterrupt! finishing..")
    interrupt = True
    #sys.exit(0)

# Register Signal Handler
signal.signal(signal.SIGINT, signal_handler)

# Main
if __name__ == '__main__':
    main()

