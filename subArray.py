

def findMaxSubArray(l):
	
	max_arr = [l[0]]
	max_summ = l[0]
	max_end = 0
	
	for i,s in enumerate(l[1:]):
		sub_summ = max(max_arr[-1] + s , s)
		max_arr.append(sub_summ)
		if sub_summ >= max_summ:
			max_summ = sub_summ
			max_end = i+1
	i = 0
	max_start = max_end
	while l[max_start] != max_arr[max_start]:
		max_start -=1
		i+=1
	return l[max_start:max_end+1]



l = [-2,1,-3,4,-1,2,1,-5,4]
print(findMaxSubArray(l))