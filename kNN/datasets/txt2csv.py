import pandas as pd

# read data from .txt file separated by '\t'
data = pd.read_csv('datingTestSet2.txt', sep='\t', header=None, 
	                names=["Flight miles per year", "Video game occupation", "Ice-cream consumption", "Type"])
print(data)
data.to_csv('E:\\GitHub\\Machine-Learning\\kNN\\datingTestSet2.csv', index=False)

#data = pd.read_csv('datingTestSet2.csv')
#print(data)