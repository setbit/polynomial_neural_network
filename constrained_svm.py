import numpy as np
from scipy import linalg
import math
from sklearn import svm


train_data=[]
test_data=[]
count=0
file=open('breast-cancer_scale','rb')
feature_size=10
for line in file:
	line=line.strip()
	Line=line.split(' ')
	Data={}
	Data['value']=int(Line[0])
	Line=Line[1:]
	for word in Line:
		Words=word.split(':')
		Data[int(Words[0])]=float(Words[1])
	if count%3==0:
		test_data.append(Data)
	else:
		train_data.append(Data)
	count+=1
#print len(test_data),len(train_data)

X_list=[]
train_output=[]
test_output=[]
for data in train_data:
	List=[1]
	for i in range(feature_size):
		if i+1 in data:
			List.append(data[i+1])
		else:
			List.append(0)
	X_list.append(List)
	train_output.append(data['value'])
train_output=np.array(train_output)
X=np.array(X_list)
train_column=X.shape[1]
m=X.shape[0]

X_list_valid=[]
for data in test_data:
	List=[1]
	for i in range(feature_size):
		if i+1 in data:
			List.append(data[i+1])
		else:
			List.append(0)
	X_list_valid.append(List)
	test_output.append(data['value'])
test_output=np.array(test_output)
X_valid=np.array(X_list_valid)

#linear
C=1
model=svm.SVC(kernel='linear',C=1,gamma='auto')
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'linear',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=10
model=svm.SVC(kernel='rbf',C=10,gamma='auto')
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'linear',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=100
model=svm.SVC(kernel='rbf',C=100,gamma='auto')
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'linear',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

#rbf
C=1
model=svm.SVC(kernel='rbf',C=1,gamma='auto')
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'rbf',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=10
model=svm.SVC(kernel='linear',C=10,gamma='auto')
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'rbf',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=100
model=svm.SVC(kernel='linear',C=100,gamma='auto')
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'rbf',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

#poly
C=1
model=svm.SVC(kernel='poly',C=1,gamma='auto',coef0=1)
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'poly',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=10
model=svm.SVC(kernel='poly',C=10,gamma='auto',coef0=1)
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'poly',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=100
model=svm.SVC(kernel='poly',C=100,gamma='auto',coef0=1)
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'poly',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

#sigmoid
C=1
model=svm.SVC(kernel='sigmoid',C=1,gamma='auto',coef0=1)
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'sigmoid',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=10
model=svm.SVC(kernel='sigmoid',C=10,gamma='auto',coef0=1)
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'sigmoid',C,incorrect,total,100*(1-(float(incorrect)/float(total)))

C=100
model=svm.SVC(kernel='sigmoid',C=100,gamma='auto',coef0=1)
train_output=np.array(train_output).reshape((len(train_output),1))
model.fit(X,train_output)
predicted=model.predict(X_valid)
#print predicted
incorrect=0.0
total=0.0
for x in range(len(predicted)):
	if predicted[x]!=test_output[x]:
		incorrect+=1
	total+=1.0
print 'sigmoid',C,incorrect,total,100*(1-(float(incorrect)/float(total)))