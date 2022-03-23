################### Numerical utilities  ######################
# Simple utils like std, median                               #
# Python version written by Piyush Agram                      #
# Date: Jan 2, 2012                                           #
###############################################################
import numpy as np
import sys

def read_rsc(fname):
    rdict = {}
    infile = open(fname,'r');
    line = infile.readline()
    while line:
        llist = line.split();
        rdict[llist[0]] = llist[1]
        line=infile.readline()

    infile.close();
    return rdict

class LineCounter:
    '''Creates a text-base line counter.'''
    def __init__(self, txt, width=30):
        self.txt = txt
        self.count = 0
        print('\n')

    def update(self, newcount):
        '''Update the counter.'''
        self.count = newcount
        strg = '%s : %8d'%(self.txt,self.count)
        sys.stdout.write('\r' + strg)
        sys.stdout.flush()

    def increment(self):
        '''Increment the counter.'''
        self.count = self.count+1
        strg = '%s : %8d'%(self.txt,self.count)
        sys.stdout.write('\r' + strg)
        sys.stdout.flush()

    def close(self):
        print('\n')


############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
