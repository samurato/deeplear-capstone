import numpy as np
import pandas as pd
import os 

dir_path = os.getcwd()
#    filename2 = dir_path + "/datasets/unsw/UNSW_NB15_testing-set.csv"
#    filename1 = dir_path + "/datasets/unsw/UNSW_NB15_training-set.csv"
filename2 =  "datasets/unsw/UNSW_NB15_testing-set.csv"
filename1 =  "datasets/unsw/UNSW_NB15_training-set.csv"

Train_datasets = pd.read_csv(filename2)
Test_datasets = pd.read_csv(filename1)

####Create a Data sets by dropping the id and attack and lables.
new_Train_datasets = Train_datasets.drop(['id','attack_cat','label'], axis = 1)
new_Test_datasets = Test_datasets.drop(['id','attack_cat','label'], axis = 1)


new_Train_labels = Train_datasets.loc[:,'attack_cat']s
new_Test_labels = Test_datasets.loc[:,'attack_cat']
convert_Train_labels = new_Train_labels.copy()
convert_Test_labels = new_Test_labels.copy()

attack_labels =new_Train_labels.unique()
print(attack_labels)
pos = np.where(attack_labels == 'Worms')
pos = pos[0]
print(pos)
attack_labels.index('Normal')

#Convert Attack LAbels to numbers for Train Labels, take the index of unique value in the list
# of attacks and convert the name of attack to the position hold by the index.
j = 0
i = 0
for j in range (len(new_Train_labels)):
    for i in range (len(attack_labels)):
        if attack_labels[i] == new_Train_labels[j]:
            convert_Train_labels[j] =i
            print (convert_Train_labels[j])

#Convert Attack Labels to unique numbers for Test Labels
j = 0
i = 0
for j in range (len(new_Test_labels)):
    for i in range (len(attack_labels)):
        if attack_labels[i] == new_Test_labels[j]:
            convert_Test_labels[j] =i
            print (convert_Test_labels[j])

#Do same thing for Training protocols
test_protocol = Test_datasets.loc[:,'proto']
test_protocol_unique = Test_datasets.loc[:,'proto'].unique()
j = 0
i = 0
for j in range (len(test_protocol)):
    for i in range (len(test_protocol_unique)):
        if test_protocol[j] == test_protocol_unique[i]:
            test_protocol[j] = i
            print (test_protocol[j],"num", j)

#Do same thing for Training protocols
protocol = Train_datasets.loc[:,'proto']
protocol_unique = Train_datasets.loc[:,'proto'].unique()

j = 0
i = 0
for j in range (len(protocol)):
    for i in range (len(protocol_unique)):
        if protocol[j] == protocol_unique[i]:
            protocol[j] = i
            print (protocol[j],"num", j)

print(protocol)

test_protocol.to_csv("Test.csv"index = False)
    






print(Train_datasets['attack'].value_counts())
print(Test_datasets['attack'].value_counts())



dataset1 = read_data(filename1)
dataset2 = read_data(filename2)

print(dataset1['attack'].value_counts())
print(dataset2['attack'].value_counts())
dataset1['proto'], dataset1['service'], dataset1['state'],  dataset2['proto'], dataset2['service'], dataset2['state'] = nominal(dataset1, dataset2)

print(dataset2['service'].value_counts())


print(dataset1['proto'].value_counts())
print(dataset2['state'].value_counts())

num_features = ["dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
"rate","sttl", "dttl","sload","dload","sloss","dloss",
"sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
"synack","ackdat","smean","dmean","trans_depth","response_body_len",
"ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
"ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
"ct_src_dst", "is_sm_ips_ports"
]

dataset1[num_features] = dataset1[num_features].astype(float)
#dataset1[num_features] = dataset1[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
#print(dataset.describe())
#dataset1[num_features] = dataset1[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
dataset1[num_features] = MinMaxScaler().fit_transform(dataset1[num_features].values)

dataset2[num_features] = dataset2[num_features].astype(float)
#dataset2[num_features] = dataset2[num_features].apply(lambda x:MinMaxScaler().fit_transform(x))
dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)

labels1 = dataset1['attack'].copy()
print(labels1.unique())


features2 = dataset2[num_features]
labels2 = dataset2['attack'].copy()


labels1[labels1 == 'Normal'] = 0
labels1[labels1 == 'Generic'] = 1
labels1[labels1 == 'Exploits'] = 2
labels1[labels1 == 'Fuzzers'] = 3
labels1[labels1 == 'DoS'] = 4
labels1[labels1 == 'Reconnaissance'] = 5
labels1[labels1 == 'Analysis'] = 6
labels1[labels1 == 'Backdoor'] = 7
labels1[labels1 == 'Shellcode'] = 8
labels1[labels1 == 'Worms'] = 9

labels2[labels2 == 'Normal'] = 0
labels2[labels2 == 'Generic'] = 1
labels2[labels2 == 'Exploits'] = 2
labels2[labels2 == 'Fuzzers'] = 3
labels2[labels2 == 'DoS'] = 4
labels2[labels2 == 'Reconnaissance'] = 5
labels2[labels2 == 'Analysis'] = 6
labels2[labels2 == 'Backdoor'] = 7
labels2[labels2 == 'Shellcode'] = 8
labels2[labels2 == 'Worms'] = 9
"""
labels1[labels1 != 'Normal'] = 1
labels1[labels1 == 'Normal'] = 0
"""
dataset1['attack'] = labels1
dataset2['attack'] = labels2
#dataset1, dataset2 = data_shuffle(dataset1 ,dataset2)
#print(dataset1['label'].value_counts())
#print(dataset2['label'].value_counts())
acc = read_data_set(dataset1, dataset2)

print(acc.train.labels)

