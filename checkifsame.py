#This program takes in three directories and checks if any of their filenames are the same
import sys
import os

dir1 = 'F:/AllData/extradata/bag1/center'
dir2 = 'F:/AllData/extradata/bag1/left'
dir3 = 'F:/AllData/extradata/bag1/right'

set1 = set(os.listdir(dir1))
set2 = set(os.listdir(dir2))
set3 = set(os.listdir(dir3))

print(len(set1.intersection(set2)))
print(len(set2.intersection(set3)))
print(len(set1.intersection(set3)))
