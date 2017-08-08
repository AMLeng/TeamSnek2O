for step in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    file = open('data/eurotruck_saves/' + str(step) + '/approximatedTimestamps.txt', 'r')
    curLine = file.readline()
    theString = ""
    while curLine is not None and not curLine == "":
        split = curLine.split(' ')
        theString = theString + (split[0] + ' ' + str(float(split[1])/180) + '\n')
        curLine = file.readline()

    file.close()
    file = open('data/eurotruck_saves/' + str(step) + '/approximatedTimestamps.txt', 'w')
    file.write(theString)