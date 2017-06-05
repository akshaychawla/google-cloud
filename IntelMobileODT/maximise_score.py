''' Maximise score by converting to 1.0 wherever possible ''' 

import csv 
import numpy as np
from math import ceil, floor
import os, sys
from copy import deepcopy

rows = []

f = open("subm_kaggledown.csv","rb")
reader = csv.reader(f)
_ = reader.next()

for row in reader:
	rows.append([row[0], float(row[1]), float(row[2]), float(row[3])])
f.close()

# ceil to one
ceiled_rows = [] 
for row in rows:
	max_val = max(row[1:])
	max_ind = row.index(max_val)
	new_row = [row[0], 0.0, 0.0, 0.0]
	new_row[max_ind] = 1.0
	

	# if abs(max_val-stripped_row[0])>=0.2 and abs(max_val-stripped_row[0])>=0.2:
	# 	max_ind = row.index(max_val)
	# 	row[max_ind] = 1.0
	ceiled_rows.append(new_row)

# print ceiled_rows

# dump to disk 
with open("subm_maxed.csv","wb") as f:
	writer = csv.writer(f)
	writer.writerow(["image_name","Type_1","Type_2","Type_3"])
	for row in ceiled_rows:
		writer.writerow(row)
	f.close()



