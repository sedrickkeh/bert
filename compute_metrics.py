def simple_accuracy(preds, labels):
	return (preds == labels).mean()

def full_accuracy(preds, labels):
	# [exactly 1 match, exactly 2 matches, exactly 3 matches, exactly 4 matches]
	num_same_array = [0,0,0,0]
	for i in range(len(preds)):
		curr_cnt = 0
		if (preds[i][0] == labels[i][0]): curr_cnt += 1
		if (preds[i][1] == labels[i][1]): curr_cnt += 1
		if (preds[i][2] == labels[i][2]): curr_cnt += 1
		if (preds[i][3] == labels[i][3]): curr_cnt += 1
		num_same_array[curr_cnt-1] += 1
	# cumulative counts 
	# [at least 1 match, at least 2 matches, at least 3 matches, at least 4 matches]
	num_same_array[2] += num_same_array[3]
	num_same_array[1] += num_same_array[2]
	num_same_array[0] += num_same_array[1]
	for i in range(len(num_same_array)):
		num_same_array[i] /= len(preds)
	return num_same_array

def compute_metrics(preds, labels):
	simple_acc = simple_accuracy(preds, labels)
	metrics = {"simple_acc": simple_acc}

	if (len(preds[0]) == 1):
		return metrics
	else:
		full_acc = full_accuracy(preds, labels)
		metrics["at_least_1_match"] = full_acc[0] 
		metrics["at_least_2_matches"] = full_acc[1] 
		metrics["at_least_3_matches"] = full_acc[2]
		metrics["all_4_matches"] = full_acc[3]
		return metrics