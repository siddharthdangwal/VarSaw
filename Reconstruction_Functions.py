from collections import Counter
import math
import numpy as np, numpy.random
import pprint
from collections import OrderedDict
import helper_functions as HF

def normalize_dict(input_dict):
    '''
    Function to normalize a dictionary
    '''

    #if the input dictionary is an empty dictionary
    if (len(input_dict.values()) == 0):
        return input_dict

    epsilon = 0.0000001 
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


def compute_entropy(in_dict):
    '''
    Function to compute the entropy of a distribution (input is a dictionary)
    '''
    norm_in = normalize_dict(in_dict.copy())
    norm_in_filtered = {key:val for key, val in norm_in.items() if val != 0.0}
    list_p = norm_in_filtered.values()
    return sum([(-i* math.log2(i)) for i in list_p])


def Compute_Helinger(dict_in,dict_ideal):
    '''
    Function to compute the Hellinger Distance between two dictionaries
    '''
    
    epsilon = 0.0000000001
    
    _in1 = Counter(normalize_dict(dict_in.copy()))
    _in2 = Counter(normalize_dict(dict_ideal.copy()))
    
    a = {}
    b = {}
    
    if list(_in1.keys()) != list(_in2.keys()):
        a = Counter(dict.fromkeys(normalize_dict(dict_in), epsilon))
        b = Counter(dict.fromkeys(normalize_dict(dict_ideal), epsilon))

    p = Counter(_in1) + Counter(b)
    q = Counter(_in2) + Counter(a)
    glob_set = set(p+q)
    
    list_of_squares = []
    for key in glob_set:

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p[key]) - math.sqrt(q[key])) ** 2

        # append 
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = math.sqrt(sum(list_of_squares)) 

    return sosq/math.sqrt(2)


def convert_binary_string(list_int):
    '''
    Args:
    list_int: A list of integers
    
    Returns
    A list of binary bitstrings for a given list of integers, where each element in the list
    has as many bits as the bitstring for the largest number in 'list_int'
    '''    
    output = []
    bitwidth = math.ceil(math.log(max(list_int)+1,2))
    bitwidth_f = str("{0:0"+ str(bitwidth) + "b}")
    
    for ele in list_int:    
        output.append(bitwidth_f.format(ele))
    
    return output


def find_binary_substring(input_dict, substring, location_list):
    '''
    Function that returns the binary substring for specific locations (qubit positions)
    '''

    '''
    What is input dict? How is it supposed to be? What is substring? What is location list?
    '''

    if sum(location_list) == 0.5*len(location_list)*(2*min(location_list)+(len(location_list)-1)):
        filtered_dict = dict(filter(lambda item: substring in item[0], input_dict.items())) 
        output_dict = {}
        for key in filtered_dict:
            temp_string = ""
            for ele in location_list:
                temp_string += key[len(key)-ele-1]
            if temp_string == substring:
                output_dict.update({key:filtered_dict[key]})
        return output_dict
    else:
        print('binary substring demands location_list as a sequence, use diffrent function!')
        return -1


def find_distrubuted_binary_substring(input_dict, substring, location_list):

    '''
    What does this function do?
    '''
    
    output_dict = {}
    for key in input_dict:
        #print('Current Key ', key)
        temp_string = ""
        for ele in location_list:
            temp_string += key[len(key)-ele-1]
        if temp_string == substring:
            output_dict.update({key:input_dict[key]})
    
    return output_dict
    
def Create_Marginals(orignal_count, marginal_order):
    '''
    Function to create a Marginal from the output histogram with all measurements and given marginal order
    ''' 

    '''
    What is a marginal order?
    '''

    norm_orignal_dict = normalize_dict(orignal_count)
    list_binary_string = convert_binary_string([*range(2**len(marginal_order))])
    output ={}
    
    if sum(marginal_order) == 0.5*len(marginal_order)*(2*min(marginal_order)+(len(marginal_order)-1)):
        #print("Serial Marginal")
        for key in list_binary_string:
            matching_dict = find_binary_substring(norm_orignal_dict,key,marginal_order) 
            val = sum(matching_dict.values())
            output.update({key:val})
    else:
        #print("Distrubuted Marginal")
        for key in list_binary_string:
            matching_dict = find_distrubuted_binary_substring(norm_orignal_dict,key,marginal_order)
            val = sum(matching_dict.values())
            output.update({key:val})
    
    return output


def marginal_circ_gen(_in, marginal_order):
    
    '''
    A few doubts in this function
    '''

    '''
    Args:
    _in: The input circuit
    marginal_order: A list consisting of the qubit

    Function to create a marginal circuit from qubit list

    Question: Why are we reversing the marginal order dictionary here??
    ''' 
    
    circ_in = _in.copy()
    circ_in.remove_final_measurements()
    creg = ClassicalRegister(len(marginal_order), name='c')
    circ_in.add_register(creg)
    circ_in.measure(marginal_order[::-1],[*range(len(marginal_order))])
    
    return circ_in


def reconstruct(baseline_counts_dict, marginal_counts_dict):
    '''
    Function to obtain the reconstruction output, given an input dictionary and a marginal
    ''' 
    _baseline = baseline_counts_dict.copy()

    _partial = marginal_counts_dict.copy()
 
    B = Amplifying_Reconstruction(_baseline, _partial)
    
    sum_marginals = Counter({})
    num_marginals = len(marginal_counts_dict)
    
    for i in range(num_marginals):
        #print('Performing Weighted Reconstruction for ', _partial[i][1]['Order'])
        #ref =  Create_Marginals(orignal_count = _baseline, marginal_order = _partial[i][1]['Order'])
        sum_marginals += Counter(B[i][0])
    
    result = normalize_dict(dict (Counter(_baseline) + sum_marginals))
    
    return result
    

def recurr_reconstruct(baseline_counts_dict, marginal_counts_dict,max_recur_count):
    '''
    Function to recursively reconstruct and terminate when the distribution does not change or when a max counter is reached
    ''' 
    _baseline = baseline_counts_dict.copy()

    _partial = marginal_counts_dict.copy()
 
    reconstructed = reconstruct(_baseline, _partial)
           
    for i in range(max_recur_count):
        _baseline = [reconstructed]
        reconstructed = reconstruct(_baseline[0], _partial)
        if Compute_Helinger(_baseline[0],reconstructed) < 0.0001:
            break
        
    return reconstructed


def Amplifying_Reconstruction (Orignal_Count, List_Marginal_Count):
    '''
    Function to readjust the probabilities of each outcome
    '''       
    norm_orignal_dict  = normalize_dict(Orignal_Count)
       
    Output_Count_List = []
    for ele in List_Marginal_Count:
        
        norm_marginal_dict  = normalize_dict(ele[0])
        copy_norm_original_dict = norm_orignal_dict.copy()
        marginal_order = ele[1]['Order']
        if sum(marginal_order) == 0.5*len(marginal_order)*(2*min(marginal_order)+(len(marginal_order)-1)):
            for key in norm_marginal_dict:
                matching_dict = find_binary_substring(copy_norm_original_dict,key,marginal_order)
                if bool(matching_dict):
                    output_dict = normalize_dict(matching_dict)
                    copy_norm_original_dict.update((x, y*(norm_marginal_dict[key]/(1-norm_marginal_dict[key]))) for x, y in output_dict.items())
        else:
            for key in norm_marginal_dict:
                matching_dict = find_distrubuted_binary_substring(copy_norm_original_dict,key,marginal_order)
                if bool(matching_dict):
                    output_dict = normalize_dict(matching_dict)
                    copy_norm_original_dict.update((x, y*(norm_marginal_dict[key]/(1-norm_marginal_dict[key]))) for x, y in output_dict.items())

        Output_Count_List.append([normalize_dict(copy_norm_original_dict),{'Order':marginal_order}])


    return Output_Count_List
