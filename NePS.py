import json
from operator import itemgetter
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

import math as mt
import csv
from sparsesvd import sparsesvd

def harmonic_sum(n):
	sum = 0
	for i in range(1,n+1):
		sum += (1.0/i)
	return sum



data = []
with open('./trunc1L.json','r') as f:  #load dataset : https://jmcauley.ucsd.edu/data/amazon/
	for line in f:
		data.append(json.loads(line))

# finding the helpfulness score for each review
for l in data:
	if l["helpful"][1]!=0 :
		l["helpfulness_score"] = l["helpful"][0]**2.0/l["helpful"][1]
	else :
		l["helpfulness_score"] = 0

# finding the rank of each review
sorted_helpful = sorted(data, key=itemgetter('helpfulness_score'), reverse=True) 

productMap = {}

for l in sorted_helpful:
	if l["asin"] in productMap:

		if productMap[l["asin"]][1] == l["helpfulness_score"]:
			l["rank"]=productMap[l["asin"]][0]
			productMap[l["asin"]][2] += 1

		else:
			productMap[l["asin"]][2] += 1
			l["rank"]=productMap[l["asin"]][2]
			productMap[l["asin"]][0] = productMap[l["asin"]][2]
			productMap[l["asin"]][1] = l["helpfulness_score"]

	else:
		productMap[l["asin"]] = [1,l["helpfulness_score"],1]
		l["rank"]=productMap[l["asin"]][0]

# finding the position of each review
sorted_time = sorted(sorted_helpful, key=itemgetter('unixReviewTime'))

productMap = {}

for l in sorted_time:
	if l["asin"] in productMap:
		productMap[l["asin"]] += 1
		l["count"]=productMap[l["asin"]]

	else:
		productMap[l["asin"]] = 1
		l["count"]=productMap[l["asin"]]

# finding the degree of each review
for l in sorted_time:
	l["degree_topmost"]=(1.0/(l["rank"]**2))*(productMap[l["asin"]]-l["count"])
	l["degree_mostrecent"]=harmonic_sum(productMap[l["asin"]]-l["count"])
	l["degree"]=l["degree_topmost"]+l["degree_mostrecent"]
	
# f = open('centrality.txt','w');
sorted_itemwise = sorted(sorted_helpful, key=itemgetter('asin'))

# for l in sorted_itemwise:
# 	print "reviewerID : %s, asin : %s, time : %s, rating : %f, helpfulness_score : %f, centrality : %f, rank : %d" %(l["reviewerID"],l["asin"],l["reviewTime"],l["overall"],l["helpfulness_score"],l["degree"],l["rank"])
	# f.write("reviewerID : %s, asin : %s, time : %s, rating : %f, helpfulness_score : %f, centrality : %f\n" %(l["reviewerID"],l["asin"],l["reviewTime"],l["overall"],l["helpfulness_score"],l["degree"]))

countU=0
countP=0
uIndex={}
pIndex={}

for l in sorted_itemwise:
	if l["reviewerID"] not in uIndex:
		uIndex[l["reviewerID"]]=countU
		countU+=1

	if l["asin"] not in pIndex:
		pIndex[l["asin"]]=countP
		countP+=1

MAX_UID = countU
MAX_PID = countP

urmRating = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)
urmHelpful = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)
urmCentrality = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)

for l in sorted_itemwise:
	urmRating[uIndex[l["reviewerID"]], pIndex[l["asin"]]] = float(l["overall"])
	urmHelpful[uIndex[l["reviewerID"]], pIndex[l["asin"]]] = float(l["helpfulness_score"])
	urmCentrality[uIndex[l["reviewerID"]], pIndex[l["asin"]]] = float(l["degree"])

# select a user,product to predict. Implement a function to find users who have purchased multiple products.
# select any one of the product and remove it.

urmCsrRating = csr_matrix(urmRating, dtype=np.float32)
urmCsrHelpful = csr_matrix(urmHelpful, dtype=np.float32)
urmCsrCentrality = csr_matrix(urmCentrality, dtype=np.float32)



f.close()
