import os, re

for item in os.listdir('.'):
    print (item)
    newname = re.sub(r"(_A_fake_A)", "", item)
    os.rename(item, newname)
    print ("-->" + newname)
