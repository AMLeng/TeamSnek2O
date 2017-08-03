import sys
import os
import re
import math

bag_num='6'
image_dir='F:/AllData/extradata/bag'+bag_num+'/center'
log_file='F:/AllData/extradata/bag'+bag_num+'steering.txt'
save_file='bag'+bag_num+'approximated.txt'

filename_list=os.listdir(image_dir)
file_stamp_list=[]
#Filter files to have images only
for possible_file in filename_list:
    if possible_file[-4:] == '.jpg':
        file_stamp_list.append(possible_file[:-4]) #If we have a valid jpg, strip the .jpg part and add to the list of valid file stamps
print(len(file_stamp_list))

#Now read in the labels
log = open(log_file,'r')
line = log.readline()
stamps = []
angles = []
while line != '':
    for i in range(2):
        log.readline()
    secs = (int) (re.sub('\D', '', log.readline()))
    nsecs = (int) (re.sub('\D', '', log.readline())[:3])
    stamp = secs + (0.001)*nsecs
    stamps.append(stamp)
    log.readline()
    angle = (re.sub('[^\d|\-|\.]','', log.readline()))
    angles.append(angle)
    for i in range(12):
        log.readline()
    line = log.readline()
save=open(save_file,'w')

for file_stamp in file_stamp_list: #Now get the closest steering angle for each filestamp
    i=0
    min_i=0
    min_diff=math.fabs(((float) (file_stamp))-stamps[min_i])
    while(i<len(stamps)):
        diff = math.fabs(((float) (file_stamp))-stamps[i])
        if diff<min_diff:
            min_i = i
            min_diff = diff
        i+=1
    save.write((file_stamp) + ' ' + str(angles[min_i]) + '\n')
