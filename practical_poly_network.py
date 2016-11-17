import numpy as np
from scipy import linalg
import math
from collections import defaultdict
import collections

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

Limited_width=10
batch_size=1
no_class=26

def BuildBasis1(FA1):
	L,D,WT=linalg.svd(FA1)
	W=[]
	WT=np.array(WT)
	D=list(D)
	D.sort(reverse=True)
	count=0
	for val in D:
		if val!=0:
			W.append(WT[count,:])
		count+=1
		if count==Limited_width:
			break
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
		if linalg.norm(Bc)!=0:
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
	#print n_output,test_output,incorrect,total,(float(incorrect)/float(total))
	print incorrect,total,(1.0-(float(incorrect)/float(total)))*100
	return n_output,(float(incorrect)/float(total))

tol=0.00000000001

def Max_Li_indices(M,r,c):
	M=np.array(M).reshape((r,c))
	b=batch_size
	c=M.shape[1]
	norm_dict=collections.defaultdict(list)
	norm_list=[]
	for i in range(c):
		norm=linalg.norm(M[:,i])
		#norm=10000000*round(norm,6)
		if norm not in norm_list:
			norm_list.append(norm)
		norm_dict[norm].append(i)
	norm_list.sort(reverse=True)
	R=np.zeros((M.shape[0],M.shape[1]))
	count=0
	Dict_index={}
	for x in norm_list:
		for z in norm_dict[x]:
			R[:,count]=M[:,z]
			Dict_index[count]=z
			count+=1
	r = np.linalg.matrix_rank(R) 
	index = [] #this will save the positions of the li columns in the matrix
	counter = 0
	index.append(0) #without loss of generality we pick the first column as linearly independent
	j = 0 #therefore the second index is simply 0

	for i in range(R.shape[1]): #loop over the columns
	    if i != j: #if the two columns are not the same
	        R1=np.array(R[:,i]).reshape((R.shape[0],1))
	        R2= np.array(R[:,j]).reshape((R.shape[0],1))
	        #print R1.shape[1], R2.shape[1]
	        inner_product = np.dot(R1.T,R2) #compute the scalar product
	        norm_i = linalg.norm(R[:,i]) #compute norms
	        norm_j = linalg.norm(R[:,j])

	        #inner product and the product of the norms are equal only if the two vectors are parallel
	        #therefore we are looking for the ones which exhibit a difference which is bigger than a threshold
	        if abs(inner_product - norm_j * norm_i) > 1e-4:
	            index.append(i) #index is saved
	            j = i #j is refreshed
	        #do not forget to refresh j: otherwise you would compute only the vectors li with the first column!!

	i = 0
	max_index=[]
	if b<r:
		R_independent=np.zeros((M.shape[0],b))
	else:
		R_independent=np.zeros((M.shape[0],r))

	while i<r and i<b:
		max_index.append(Dict_index[index[i]])
		R_independent[:,i]=R[:,index[i]]
		i+=1

	return max_index,R_independent


def BuildBasist(F,FAT,Node_List):
	FT=[]
	W=[]
	O=linalg.orth(F)
	Pos=[]
	count=0
	V=np.array(train_output).reshape((len(train_output),1))
	# V=np.zeros((len(train_output),no_class))
	# for i in range(len(train_output)):
	# 	V[i,train_output[i]-1]=1
	O=np.array(O)
	OP=np.inner(O,O)
	OP=np.array(OP)
	prod=np.dot(OP,V)
	prod=np.array(prod)
	V=np.subtract(V,prod)
	if FAT.shape[1]!=len(Node_List):
		print FAT.shape[1],len(Node_List)
	for i in range(Limited_width/batch_size):
		O=np.array(O)
		OP=np.inner(O,O)
		OP=np.array(OP)
		prod=np.dot(OP,FAT)
		prod=np.array(prod)
		C=np.subtract(FAT,prod)
		C=np.array(C)
		norm=linalg.norm(C)
		OV=linalg.orth(V)
		OV=OV.T
		for i in range(C.shape[1]):
			norm=linalg.norm(C[:,i])
			if norm!=0:
				norm=1/norm
				C[:,i]=np.multiply(C[:,i],norm)
		OV_prod=np.mat(OV)*np.mat(C)
		index,BAS=Max_Li_indices(list(OV_prod),OV_prod.shape[0],OV_prod.shape[1])
		C=np.array(C)
		C_mat=np.zeros((C.shape[0],batch_size))
		indice=0
		for val in index:
			norm=linalg.norm(FAT[:,val])
			if norm!=0:
				norm=math.sqrt(m)/norm
			N=FAT[:,val]
			N=np.array(N).reshape((FAT.shape[0],1))
			N=np.multiply(N,norm)
			N=np.array(N).reshape((FAT.shape[0],1))
			C_mat[:,indice]=C[:,val]
			indice+=1
			if count==0:
				FT=N
				FT=np.reshape(FT,(FAT.shape[0],1))
				count+=1
			else:
				FT=np.concatenate((FT,N),axis=1)
				FT=np.array(FT)
				count+=1
			W.append(norm)
			Pos.append(Node_List[val])
		#print O.shape[0],O.shape[1]
		OC=linalg.orth(C_mat)
		OC=np.array(OC)
		O=np.concatenate((O,OC),axis=1)
		OC_prod=np.inner(OC,OC)
		OC_prod=np.array(OC_prod)
		OC_prod=np.multiply(OC_prod,V)
		OC_prod=np.array(OC_prod)
		V=np.subtract(V,OC_prod)
		V=np.array(V)
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
for i in xrange(2,10):
	(n_output,error)=outputLayer(F,Validation_Node)
	print error,i
	if error<thresold:
	 	#print n_output
	 	break
	count=0
	FA=np.zeros((FT.shape[0],FT.shape[1]*F1.shape[1]))
	#FA=[]
	Node_List=[]
	for i in range(FT.shape[1]):
		for j in range(F1.shape[1]):
			F1c=F1[:,j]
			Fc=FT[:,i]
			FA[:,count]=np.multiply(F1c,Fc)
			# if count==0:
			# 	FA=np.multiply(F1c,Fc)
			# 	FA=np.array(FA).reshape((FA.shape[0],1))
			# else:
			# 	mult=np.array(np.multiply(F1c,Fc)).reshape((FA.shape[0],1))
			# 	FA=np.concatenate((FA,mult),axis=1)
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