import numpy as np
from scipy import linalg
import math

train_data=[]
test_data=[]
count=0
file=open('breast-cancer_scale','rb')
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
	for i in range(10):
		List.append(data[i+1])
	X_list.append(List)
	train_output.append(data['value'])
train_output=np.array(train_output)
X=np.array(X_list)
train_column=X.shape[1]
m=X.shape[0]

X_list_valid=[]
for data in test_data:
	List=[1]
	for i in range(10):
		List.append(data[i+1])
	X_list_valid.append(List)
	test_output.append(data['value'])
test_output=np.array(test_output)
X_valid=np.array(X_list_valid)


def BuildBasis1(FA1):
	L,D,WT=linalg.svd(FA1)
	W=[]
	WT=np.array(WT)
	count=0
	for val in D:
		if val!=0:
			W.append(WT[count,:])
		count+=1
	W=np.array(W)
	#print W.shape[0],W.shape[1]
	W=W.T
	#print W.shape[0],W.shape[1]
	B=np.dot(FA1,W)
	c=W.shape[1]
	#m=FA1.shape[0]
	for i in range(c):
		Bc=B[:,i]
		Wc=W[:,i]
		b=math.sqrt(m)/linalg.norm(Bc)
		for x in range(len(Bc)):
			Bc[x]=b*Bc[x]
		for y in range(len(Wc)):
			Wc[y]=b*Wc[y]
		B[:,i]=Bc
		W[:,i]=Wc
	return (B,W)



def outputLayer(F,Validation_Node):
	F_train=F
	w,res,rank,singular=linalg.lstsq(F_train,train_output)
	w=np.array(w)
	Node_test=np.array(Validation_Node)
	w=np.array(w)
	#print Node_test.shape[0],Node_test.shape[1]
	n_output=np.inner(w,Node_test.T)
	L=list(n_output)
	#print Node_test.shape[0],Node_test.shape[1]
	for x in range(len(L)):
		if L[x]<3:
			L[x]=2
		else:
			L[x]=4
	incorrect=0.0
	total=0.0
	for x in range(len(L)):
		if L[x]!=test_output[x]:
			incorrect+=1
		total+=1.0
	print incorrect,total,100*(1-(float(incorrect)/float(total)))
	return n_output,(float(incorrect)/float(total))

tol=0.00000000001

def BuildBasist(F,FAT,Node_List):
	FT=[]
	W=[]
	O=linalg.orth(F)
	Pos=[]
	count=0
	if FAT.shape[1]!=len(Node_List):
		print FAT.shape[1],len(Node_List)
	for i in range(FAT.shape[1]):
		Fc=FAT[:,i]
		O=np.array(O)
		OP=np.inner(O,O)
		OP=np.array(OP)
		prod=np.dot(OP,Fc)
		prod=np.array(prod)
		C=np.subtract(Fc,prod)
		C=np.array(C)
		norm=linalg.norm(C)
		Fc_norm=linalg.norm(Fc)
		if norm>tol:
			Fc_norm=math.sqrt(m)/Fc_norm
			for x in range(len(Fc)):
				Fc[x]=Fc_norm*Fc[x]
			Fc=np.reshape(Fc,(len(Fc),1))
			if count==0:
				FT=Fc
				FT=np.reshape(FT,(Fc.shape[0],1))
				count+=1
			else:
				FT=np.concatenate((FT,Fc),axis=1)
				FT=np.array(FT)
				count+=1
			W.append(Fc_norm)
			Pos.append(Node_List[i])
			C=np.multiply(C,1/norm)
			C=np.reshape(C,(C.shape[0],1))
			O=np.concatenate((O,C),axis=1)
	return FT,W,Pos

FA1=X
Node=[]
Validation_Node=[]
F1,W1=BuildBasis1(FA1)
F=F1
thresold=float(0.0001)
for i in range(F1.shape[1]):
	Wc=W1[:,i]
	n=np.inner(Wc,X)
	n_validation=np.inner(Wc,X_valid)
	Node.append(n)
	Validation_Node.append(n_validation)
N1=Node
N_Last=N1
FT=F1
N1_validation=Validation_Node
N_Last_validation=N1_validation
for i in xrange(2,5):
	(n_output,error)=outputLayer(F,Validation_Node)
	print i
	if i==4:
		break
	if error<thresold:
	 	#print n_output
	 	break
	count=0
	FA=np.ones((FT.shape[0],FT.shape[1]*F1.shape[1]))
	Node_List=[]
	for i in range(FT.shape[1]):
		for j in range(F1.shape[1]):
			F1c=F1[:,j]
			Fc=FT[:,i]
			FA[:,count]=np.multiply(F1c,Fc)
			count+=1
			que=[]
			que.append(i)
			que.append(j)
			Node_List.append(que)
	FT,W,Pos=BuildBasist(F,FA,Node_List)
	FT=np.array(FT)
	if FT.shape[1]==0:
		break
	node=[]
	node_validation=[]
	for x in range(len(Pos)):
		n_out=W[x]*N_Last[Pos[x][0]]*N1[Pos[x][1]]
		n_out_validation=W[x]*N_Last_validation[Pos[x][0]]*N1_validation[Pos[x][1]]
		node.append(n_out)
		node_validation.append(n_out_validation)
	N_Last=node
	N_Last_validation=node_validation
	Node.extend(node)
	Validation_Node.extend(node_validation)
	F=np.concatenate((F,FT),axis=1)
	F=np.array(F)