from __future__ import division
import re
import datetime
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os

def normalize_dict(input_dict):
    '''
    Function to normalize a dictionary
    '''
    epsilon = 0.0000001

    #if the dictioanary has zero total elements
    if sum(input_dict.values()) == 0:
        ##print('Error, dictionary with total zero elements!!')    
        for k,v in input_dict.items():
            input_dict[k] = epsilon
    factor=1.0/sum(input_dict.values())
    #if(factor == 0):
    #    print(factor,sum(input_dict.values())) 
    for k in input_dict:
        input_dict[k] = input_dict[k]*factor

    for k,v in input_dict.items():
        if(v==1):
            input_dict[k] = 1-epsilon

    return input_dict


def update_dist(dict1,dict2):
    '''
    Function to merge two dictionaries in to a third one
    '''
    dict3 = Counter(dict1) + Counter(dict2) 
    dict3 = dict(dict3)
    return dict3

def weighted_update_dist(dict1,dict2,weight):
    '''
    Function to merge two dictionaries in to a third one using a weight factor- useful for weighted EDM
    '''
    _dict2 = dict2.copy()
    for key, value in _dict2.items():
        _dict2[key] = value*weight
    dict3 = Counter(dict1) + Counter(_dict2)
    dict3 = dict(dict3)
    return dict3

def get_number_of_trials(num_qubits):
    '''
    Function to estimate total number of trials to be executed
    '''
    state_space = int(math.pow(2,num_qubits))
    num_trials = int(55.26*state_space)
    return num_trials

def read_qasm(filepath, verbo):
    '''
    Function to read a QASM into a Quantum circuit object
    '''
    circ = QuantumCircuit.from_qasm_file(filepath)
    if(verbo):
        circ.draw()
        print(circ)
        list_ops = circ.size()
        print("Total Number of Operations", list_ops)
        print('Circuit Depth: ', circ.depth())
        print('Number of Qubits in program:', circ.width())
    return circ

def write_qasm_file_from_qobj(output_file,qobj):
    '''
    Function to write a Quantum Object into a given output file QASM
    '''
    f= open(output_file,"w+")
    f.write(qobj.qasm())
    f.close()

def get_counts_given_key(counts,search_key):
    '''
    Function to get the counter value for a given key in a distribution, if not found, return 0
    This function is useful when the errors are too large and the distribution does not contain the given key
    '''
    for k,v in counts.items():
        if (k == search_key):
            return v
    return 0

def find_top_K_keys(distribution,K):
    '''
    Function to create a shortlisted candidates of top K keys
    '''
    sorted_histogram = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    key_storage = []
    occurence = []
    for i in range(K):
        key_storage.append(sorted_histogram[i][0])
        occurence.append(sorted_histogram[i][1])
        
    return key_storage,occurence
               
def convert_key_to_decimal(string, width):
    '''
    Function to convert a key to decimal
    '''
    power = width-1;
    dec_key = 0
    for c in string: # go through every character
        dec_key = dec_key + np.power(2,power)*int(c)
        power = power -1
    return dec_key

def convert_integer_to_bstring(num):
    '''
    Function to convert an integer into bitstring
    '''
    bstring = ''
    flag = 0
    if(num>1):
        bstring = convert_integer_to_bstring(num // 2)
    bstring = bstring+ str(num % 2)
    return bstring

def padding_for_binary(bstring, expected_length):
    '''
    Function to pad a bitstring with 0s and stretch it to a given length
    '''
    curr_length = len(bstring)
    if(expected_length > curr_length):
        diff_length = expected_length - curr_length
        padding = ''
        for i in range(diff_length):
            padding = padding + str(0)
        bstring = padding + bstring
    return bstring

def get_key_from_decimal(num,length):
    '''
    Function to convert a key of given length to a decimal
    '''
    bstr = convert_integer_to_bstring(num)
    key = padding_for_binary(bstr, length)
    return key

def hdist_compute(p1,p2):
    epsilon = 0.0000000001

    _in1 = Counter(normalize_dict(p1.copy()))
    _in2 = Counter(normalize_dict(p2.copy()))

    a = {}
    b = {}

    if list(_in1.keys()) != list(_in2.keys()):
        a = Counter(dict.fromkeys(normalize_dict(p1), epsilon))
        b = Counter(dict.fromkeys(normalize_dict(p2), epsilon))

    p = Counter(_in1) + Counter(b)
    q = Counter(_in2) + Counter(a)
    glob_set = set(p+q)

    list_of_squares = []
    db = 0
    for key in glob_set:
        db = db + np.sqrt(p[key]*q[key])
    bc = db
    if(np.log(db)==0):
        db = np.log(db)
    else:
        db = -1*np.log(db)
    hd = np.sqrt(1-bc)
    return hd

def compute_bhattacharyya_hellinger_distance(p1, p2):
    '''
    Function to compute hellinger distance between two distributions
    '''
    db = 0
    for i in range(len(p1)):
        db = db + np.sqrt(p1[i]*p2[i])
    bc = db # only the bhattacharyya coefficient
    if(np.log(db)==0):
        db = np.log(db)
    else:
        db = -1*np.log(db)
    hd = np.sqrt(1-bc)
    return hd

def generate_all_possible_combinations(a):
    '''
    This function returns all possible combinations of a given list (too slow for large lists)
    '''
    if len(a) == 0:
        return [[]]
    cs = []
    for c in generate_all_possible_combinations(a[1:]):
        cs += [c, c+[a[0]]]
    return cs

def xor_c(a, b):
    '''
    Function to return the XOR value of two bits
    '''
    return '0' if(a == b) else '1';

def flip(c):
    '''
    Helper function to flip the bit
    '''
    return '1' if(c == '0') else '0';

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def listinlist(l1,l2):
    '''
    Function that checks if a list l1 is present in a list of lists l2 and the entries are in same order
    '''
    if(len(l2)==0):
       return 0
    for list_entry in l2:
        match = 1 # match found
        for ele in range(len(l1)):
            if(l1[ele]!=list_entry[ele]):
                match = 0
                break
        if(match ==1):
            return match
    return match
