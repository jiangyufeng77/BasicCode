import os, re

for item in os.listdir('.'):
    if (re.match(r"^\d+", item)):
        print (item)
        newname = re.sub(r"(_A_fake_A)", "", item)
        os.rename(item, newname)
        print ("-->" + newname)
