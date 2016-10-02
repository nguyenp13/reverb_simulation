#!/usr/bin/python

import sys
import os
import ntpath
import math
import pdb
import random
import string
import xlrd
import matplotlib
import matplotlib.pyplot
import urllib
import urllib2
import re

UNIQUE_IDENTIFIER_COUNTER = 0
DEBUG=True
inf = float('inf')

def find_string(head, tail, text_to_search):
    match = re.search('('+head+').*('+tail+')', text_to_search)
    return match.group(0) if match else None

def find_all_strings(head, tail, text_to_search):
    matches = re.finditer('('+head+').*('+tail+')', text_to_search)
    ans = []
    for match in matches:
        if match:
            ans.append(match.group(0))
    return ans

def get_contents_from_link(link):
    return urllib2.urlopen(link).read()

def download_via_http(link, destination_file):
    urllib.URLopener().retrieve(link, destination_file)

def join_paths(l):
    return reduce(os.path.join,l)

def list_dir_abs(basepath):
    return map(lambda x: os.path.abspath(os.path.join(basepath, x)), os.listdir(basepath))

def plot(ax, x, y, labels0=None):
    ax.scatter(x, y,zorder=10)
    labels=['']*len(y) if labels0==None else labels0
    for i in range(len(x)):
        ax.annotate(labels[i], (x[i], y[i]))

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def divide(top,bottom):
    # Like normal division, except interprets division by zero to be inf
    if bottom==0.0:
        return inf*top
    return top/float(bottom)

def list_intersection(l1,l2):
    return filter(set(l1).__contains__, l2)

def generate_unique_identifier():
    global UNIQUE_IDENTIFIER_COUNTER 
    UNIQUE_IDENTIFIER_COUNTER += 1
    return UNIQUE_IDENTIFIER_COUNTER

def remove_file_extension(file_name):
    index_of_last_dot = file_name.rfind('.')
    return file_name if index_of_last_dot == -1 else file_name[:index_of_last_dot]

def generate_unique_file_name(extension0, placement_directory='.'):
    extension = extension0.replace('.','')
    new_file_name = None
    file_exists = True
    while file_exists:
        length_of_file_name = max(10,random.randint(0,255)-len(extension)-1) # the minus 1 is for the '.'
        new_file_name = ''.join([random.choice(string.ascii_letters) for i in xrange(length_of_file_name)])+'.'+extension
        file_exists = os.path.exists(os.path.join(placement_directory,new_file_name))
    return new_file_name

def generate_unique_directory_name(placement_directory='.'):
    new_directory_name = None
    directory_exists = True
    while directory_exists:
        length_of_directory_name = random.randint(10,255) 
        new_directory_name = ''.join([random.choice(string.ascii_letters) for i in xrange(length_of_directory_name)])
        directory_exists = os.path.exists(os.path.join(placement_directory,new_directory_name))
    return new_directory_name

def get_command_line_param_val_default_value(args, param_option, default_value):
    if param_option in args:
        param_val_index = 1+args.index(param_option)
        if param_val_index < len(args):
            return args[param_val_index]
    return default_value

def assertion(condition, message, error_code=1):
    if DEBUG:
        if not condition:
            print >> sys.stderr, ''
            print >> sys.stderr, message
            print >> sys.stderr, ''
            sys.exit(error_code)

def dict_pretty_print(d):
    max_k_len = 30
    max_v_len = 50    
    for k,v in d.items():
        max_k_len = max(max_k_len, len(str(k)))
#        max_v_len = max(max_v_len, len(str(v)))
    for k,v in sorted(d.items()):
        print ("%-"+str(max_k_len)+"s %"+str(max_v_len)+"s") % (str(k), str(v))
    print 
