import sys
import math

pred_file = open(sys.argv[1],'r')
answer_file = open(sys.argv[2],'r')

preds = []
answers = []

for line in pred_file:

	if "Id,plays" in line:
		continue 
	
	pred = float(line.rstrip().split(',')[1])
	preds.append(pred)

for line in answer_file:

	answer = float(line.rstrip())
	answers.append(answer)

if len(preds) != len(answers):
	print "lists not equal length"
else:

	total = 0
	for i in range(len(preds)):
		total += math.fabs(preds[i]-answers[i])

	print total/len(preds)
	


