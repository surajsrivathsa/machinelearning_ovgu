#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 8 21:25:24 2019

@group: Monday 13-15 pm Group 4

@Tutor name: Anirban Saha

@Authors: Chandan Radhakrishna(229746), Rashmi Koparde(230322), Suraj Shashidhar(230052)



Description: The program has to parse input data along with the expected results.
    It should come up with its own implementation of ID3 Decision tree.
    It should calculate the appropriate entropies and find out best attribute to split the data.
    
    The program should be able to run on any dataset and should be able to take 
    input and output path from the command line.
    

Run_Format: 
python3 decisiontree.py --data "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/decisiontree/nursery.csv" --output "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/decisiontree/p02_out_nursery.xml" 

python3 decisiontree.py --data "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/decisiontree/car.csv" --output "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/decisiontree/p02_out_car.xml"

"""

import lxml
import os
from lxml import etree as etree
import pandas as pd
import numpy as np
import argparse
import math
from ast import literal_eval
import os
from pathlib import Path

"""
Importing  Importing pandas for file I/O grouping and filtering, Import argparse for argument parsing
Importing lxml for xml creation and manipulation, Import ast.string_eval to convert list to strings and strings to list
Rest of the libraries are for utilities
"""

ap = argparse.ArgumentParser()

# Add the arguments to the parser and parse the arguments from command line
ap.add_argument( "--data", required=True, help="first operand")
ap.add_argument("--output", required=True, help="second operand")
args = vars(ap.parse_args())

filepath = args["data"]
output_filepath = args["output"]

"""
Function name: readInputFile
input: filepath
output: dataframe with features and expected value , with number of weights
"""
def readInputFile(filepath):
    tmpDf = pd.read_csv(filepath);
    print("Record count is: " + str(tmpDf.shape[0]))
    numOfCols = tmpDf.shape[1]
    collist = []
    base_columnname = "att"  
    t = base_columnname
    
    for i in range(0,numOfCols-1):
        t = "att"
        t = t + str(i)
        collist.append(t)
        
    collist.append("expected_value")
    
    print(*collist)
    tmpDf = pd.read_csv(filepath,  index_col = None, names = collist);
    attribute_list = collist[0:len(collist)-1]
    valuecount = tmpDf["expected_value"].value_counts().to_dict()
    numberOfClasses = 0
    
    for key,value in valuecount.items():
        numberOfClasses = numberOfClasses + 1
    
    return numberOfClasses, valuecount, attribute_list, collist, numOfCols-1, tmpDf;

# get the input dataframe along with number of weights need to be calculated and initialize the query variable
numberOfClasses, valuecount, attribute_list, collist, numOfAttributes, inputDf = readInputFile(filepath)
query = ""


#function to calculate Entropy of one attributes one value
def calculate_entropy_of_one_value_of_attribute(class_dict, number_of_classes):        
        count_of_all_points = 0;
        entropy = 0.0
        proportion = 0.0
        for key,value in class_dict.items():
            count_of_all_points = count_of_all_points + value
        
        for key,value in class_dict.items():
            proportion = value/count_of_all_points
            proportion = -1.0 * proportion * math.log(proportion, number_of_classes)
            entropy = entropy + proportion
        
        return entropy;


#Returns proportion of the records sent from parent to child
# for example if parent 1000 records are divided as 250,250 and 500 records among child then
# proportion dictionary will have {0.25, 0.25, 0.5}
def calculate_proportion(df,attribute):
    proportion_dict = {}
    xDf = df.groupby([attribute, "expected_value"]).size().reset_index(name="counts")
    distinct_class_dict = xDf["expected_value"].value_counts().to_dict()
    distinct_attribute_values_dict = xDf[attribute].value_counts().to_dict()
    proportionDf = xDf.groupby([attribute]).sum().reset_index()
    total = proportionDf["counts"].sum()
    proportionDf["counts"] = proportionDf["counts"]/total
    for index, row in proportionDf.iterrows(): 
        proportion_dict[row[attribute]] = row["counts"]
    return xDf, distinct_class_dict, proportion_dict, distinct_attribute_values_dict

""" Function to calculate the entropy of a particular attribute
1) Just pass the number of classes, attribute and distinct_attribute_values_dict which will have dis
    tinct values of attribtes in it for dynamic query generation later example take column wind: {normal: 1, high: 1, low: 1},
    Here we need only keys and not values as the values will always be 1 since they are distinct
2) For each value formualte a dynamic query on the fly and apply it on the input dataframe
    Example: previous query = "expectd_value != ''"  :: attribute = "att_5" :: key = "vhigh"
    current query = previous query + ' & ' + attribute + "'" + key + "'"
    so current query = "expectd_value != '' & att_5 == vhigh", Use this to generate required dataset for calculation on fly

3) For each value of the attrbute calculate the individual entropy of that value and return back the dictionary of entropies of individual values.
    Since we need to use these individual values later on, we need to store them in some datastructure for example dictionary
"""
def calculate_entropy(df, query, attribute, proportion_dict, number_of_classes, distinct_attribute_values_dict):
    entropy_dict = {}
    tmp_query = {}
    tmp_df = {}
    query_lst = []
    for key, value in distinct_attribute_values_dict.items():
        tmp_query[key] = attribute + " == "  +  "'" + key + "'"
        query_lst.append(tmp_query[key])
        xDf = df.query(tmp_query[key])
        xDf = xDf.groupby(["expected_value"]).size().reset_index(name="counts")
        tmp_df[key] = xDf
        #print(xDf.head(5))
        x_dict = {}
        for index,row in xDf.iterrows():
            x_dict[row["expected_value"]] = row["counts"]
        entropy_dict[key] = calculate_entropy_of_one_value_of_attribute(x_dict, number_of_classes)
    
    return x_dict, entropy_dict, tmp_query
        

def combined_child_entropy(proportion_dict,entropy_dict):
    child_entropy = 0.0
    for key, value in proportion_dict.items():
        child_entropy = child_entropy + value * entropy_dict[key]
        
    return child_entropy;

"""Function to find the best attribute according to ID3 algorithm
1) First get the entropy of parent and list of available attributes that we can split on
2) For each attribute has some distinct labels, get the distinct values available in each attribute example: (outlook = [sunny, rainy, outcast])
3) calculate_proportion: Calculate the proportion of records ie; count_of_data(filter outlook = sunny)/count_of_all_records_from_parent for each value and store it in proportion dict.
    Proportion dict is use ful when finding I(G) where we need to multiply the proportion of each value with its individual entropy Summation((Vsunny/Vall) * ENTROPYsunny + (Voutcast/Vall) * ENTROPYoutcast) + (Vrainy/Vall) * ENTROPYrainy)
    here (Vsunny/Vall), (Voutcast/Vall) and (Vrainy/Vall) are from proportion dict
4) calculate_entropy: For each inidividual value such as sunny, rainy and outcast calculate individual entropy and store it in a dictionary
5) combined_child_entropy : Now we have individual entropy and proportion entropy, to find I(G) multiply both dictionary values and sum it up
6) After you have total entropy of entire attribute, find information gain and check if it is better than that of previous attribute. 
    Finally find out which is the best attribute to split on based on info gain and take its data for next step of recursion
    
"""
def find_best_attribute(parentnode, attribute_list, inputDf, query, numberOfClasses):
    parent_entropy = float(parentnode.attrib["entropy"])
    information_gain = 0.0; req_att = ""; req_distinct_classes_dict = {}; req_distinct_attribute_values_dict = {};
    req_entropy_dict = {};req_child_entropy = 0.0; req_df = ""; req_query_dict = {};
    query_dict = {} ;
    for att in attribute_list:
        tmp_query = query
        tmp_df = inputDf.query(query)
        xDf, distinct_class_dict, proportion_dict,distinct_attribute_values_dict = calculate_proportion(tmp_df, att)
        x_dict, entropy_dict, query_dict = calculate_entropy(tmp_df, query, att, proportion_dict, numberOfClasses, distinct_attribute_values_dict)
        child_entropy = combined_child_entropy(proportion_dict,entropy_dict)
        
        if(abs(parent_entropy-child_entropy) > information_gain):
            information_gain = abs(parent_entropy-child_entropy)
            req_att = att; req_distinct_classes_dict = distinct_class_dict; req_entropy_dict = entropy_dict;
            combine_child_entropy = child_entropy; req_df = tmp_df; req_query_dict = query_dict; req_distinct_attribute_values_dict = distinct_attribute_values_dict;
        
    
    return req_att, req_entropy_dict, combine_child_entropy, information_gain, distinct_attribute_values_dict, query_dict;


"""Function to add a child to parent node in xml
1) If entropy is 0 then we need to add the text of the expected value we are getting
2) Please note that we are always doing copy by value for attribute list as this is important.
    Wthout which each node cannot have its own version of the attribute list it can split on(if it is copy by reference as every node tries to delete from same attribute list)
3) Convert the list back to string for xml node attribute addition as xml won't take any other value than string.
    So every time we do an operation on such stringified lists from xml node we need to convert the string to list -->  do operation --> reconvert it back to string and pass it on to next node
"""
def addnode(parentnode, req_att, entropy, combined_child_entropy, information_gain, attribute_value, query, node_text, attribute_list):
    
    tmp_list = attribute_list.copy();
    if req_att in tmp_list:
        tmp_list.remove(req_att)
    str_tmp_list = str(tmp_list)
    
    if(entropy > 0.0):
        new_element = etree.Element("node", entropy=str(entropy) ,feature=req_att, value=attribute_value, query = query, combined_child_entropy = str(combined_child_entropy), attribute_list = str_tmp_list)
        parentnode.append(new_element)
    elif(entropy == 0.0):
        new_element = etree.Element("node", entropy=str(entropy) ,feature=req_att, value=attribute_value, query = query, combined_child_entropy = str(combined_child_entropy), attribute_list = str_tmp_list)
        new_element.text = node_text
        parentnode.append(new_element)        
    return;

#function to convert the attribute list to str and viceversa as it is passed down to each and every node    
def get_atribute_val(df, query):
    tdf = df.query(query);
    t_list = tdf["expected_value"].unique().tolist()
    return t_list[0]

#Function to initialize the tree and the query
def initialize_tree_and_params(valuecount, numberOfClasses):
    query = "expected_value != ''"
    entropy = calculate_entropy_of_one_value_of_attribute(valuecount, numberOfClasses)
    root = etree.Element("tree",  entropy=str(entropy),entropy_of_current_node=str(entropy), \
                         combined_entropy=str(entropy), query = query, attribute_list = str(attribute_list))
    return root, query


def recursive_fnc(node, attribute_list, inputDf, query, numberOfClasses):
    query = node.attrib["query"]
    req_att, req_entropy_dict, combined_child_entropy, information_gain,  distinct_attribute_values_dict, query_dict = \
    find_best_attribute(node, attribute_list, inputDf, query, numberOfClasses)
    

    #add a condition to check if the list is null and terminate, if list is null then this means we have exhausted all attributes
    if(len(attribute_list) == 0):
        return;
    
    #dd = dict(sorted(req_entropy_dict.items(), key=lambda x: x[1]))
    #req_entropy_dict = dd
    
    for key,value in req_entropy_dict.items():
        myquery = query +  " & " +  req_att + " == " +  "'" + key + "'"
        if(req_entropy_dict[key] > 0.0):
            addnode(node, req_att, req_entropy_dict[key], combined_child_entropy, information_gain, key, myquery, "", attribute_list)
        elif(req_entropy_dict[key] == 0.0):
            node_text = get_atribute_val(inputDf, myquery)
            addnode(node, req_att, req_entropy_dict[key], combined_child_entropy, information_gain, key, myquery, node_text, attribute_list)

   #For each and every node remove the splitting attribute from attribute list and recursivelt call the function again with new query
    for child in node:
        #recursive condition
        if(float(child.attrib["entropy"]) > 0.0):
            my_tmp_list = literal_eval(child.attrib["attribute_list"])
            recursive_fnc(child,my_tmp_list, inputDf, query, numberOfClasses)        
        else:
            #Base condition
            return;
    return

#Get the initial root node with initial query
root, query = initialize_tree_and_params(valuecount, numberOfClasses)

#Call the recursive function to add nodes to the xml and calculate the decision tree
recursive_fnc(root, attribute_list, inputDf, query, numberOfClasses)


#Strip away the non essential attributes such as query etc from our tree as we need not print them in our final xml
my_tree = lxml.etree.strip_attributes(root, "combined_entropy", "combined_child_entropy", "entropy_of_current_node", "query", "attribute_list")


#Dump the etree onto an xml file specified from command line
with open(output_filepath, 'wb') as doc:
   doc.write(etree.tostring(root, pretty_print = True))



"""


output_filename = Path(filepath).resolve().stem
output_filename = "p02_out_" + output_filename + ".xml"
output_filepath = output_filepath + output_filename


with open(output_filepath, 'wb') as doc:
   doc.write(etree.tostring(root, pretty_print = True))
   

new_tree = lxml.etree.parse(output_filepath)
new_root = new_tree.getroot()

page = open("/Users/surajshashidhar/Desktop/planes.xml").read()

root = lxml.etree.parse("/Users/surajshashidhar/Desktop/planes.xml");
print(lxml.etree.tostring(root))


root.append(new_element)

new_element = etree.Element("node", entropy="0.6075789953970611" ,feature="att5" ,value="med")
root.append(new_element)

new_element = etree.Element("node", entropy="0.8077559247898062", feature="att5", value="high")
root.append(new_element)

new_element = etree.Element("node", entropy="0.8077559247898062", feature="att5", value="high")
node.append(new_element)







"""


























