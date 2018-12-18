with open("req.txt") as f:
	lis=[]
	f.readline()
	for line in f:
		line=line.split()
		lis.append(line[0]+"=="+line[1])
	print(len(lis))
