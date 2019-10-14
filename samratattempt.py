import numpy as np
import pandas as pd
import os
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from dbn.tensorflow import SupervisedDBNClassification

#import sklearn.preprocessing import OneHotEncoder
dir_path = os.getcwd()
#    filename2 = dir_path + "/datasets/unsw/UNSW_NB15_testing-set.csv"
#    filename1 = dir_path + "/datasets/unsw/UNSW_NB15_training-set.csv"
filename2 =  "datasets/unsw/UNSW_NB15_testing-set.csv"
filename1 =  "datasets/unsw/UNSW_NB15_training-set.csv"

Train_datasets = pd.read_csv(filename2)
Test_datasets = pd.read_csv(filename1)
attack_dict = {'Normal':1, 'Backdoor':2, 'Analysis':3, 'Fuzzers':4, 'Shellcode':5, 'Reconnaissance':6,'Exploits':7, 'DoS':8, 'Worms':9,'Generic':10}
Train_datasets.attack_cat =[attack_dict[item] for item in Train_datasets.attack_cat]
Test_datasets.attack_cat = [attack_dict[item] for item in Test_datasets.attack_cat]
Train_datasets = Train_datasets[0:10000]
Test_datasets = Test_datasets[0:10000]

####Create a Data sets by dropping the id and attack and lables.
new_Train_datasets = Train_datasets.drop(['id','attack_cat','label'], axis = 1)
new_Test_datasets = Test_datasets.drop(['id','attack_cat','label'], axis = 1)
new_Train_labels = Train_datasets.loc[:,'attack_cat']
new_Test_labels = Test_datasets.loc[:,'attack_cat']


#new_Train_datasets = Train_datasets[0:100]
#Protocol (proto) one hot encoding
encoded = pd.concat([new_Train_datasets, pd.get_dummies(new_Train_datasets['proto'], prefix='proto')], axis=1)
encoded.drop(['proto'], axis=1, inplace=True)

encoded_test = pd.concat([new_Test_datasets, pd.get_dummies(new_Test_datasets['proto'], prefix='proto')], axis=1)
encoded_test.drop(['proto'], axis=1, inplace=True)

#Service (service) one hot encoding
encoded = pd.concat([encoded, pd.get_dummies(new_Train_datasets['service'], prefix='service')], axis=1)
encoded.drop(['service'], axis=1, inplace=True)

encoded_test = pd.concat([encoded_test, pd.get_dummies(new_Test_datasets['service'], prefix='service')], axis=1)
encoded_test.drop(['service'], axis=1, inplace=True)

#State (state) one hot encoding
encoded = pd.concat([encoded, pd.get_dummies(new_Train_datasets['state'], prefix='state')], axis=1)
encoded.drop(['state'], axis=1, inplace=True)

encoded_test = pd.concat([encoded_test, pd.get_dummies(new_Test_datasets['state'], prefix='state')], axis=1)
encoded_test.drop(['state'], axis=1, inplace=True)
# MinMax Scaling
X_train, X_test, y_train, y_test = train_test_split(encoded,new_Train_labels, test_size = 0.20, random_state = 42)
scaler = StandardScaler

scaler = MinMaxScaler(feature_range=(0,1))
scalerb=MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scalerb.fit_transform(y_train)
y_test = scalerb.transform(y_test)

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)



classifier = SupervisedDBNClassification(hidden_layers_structure=[200, 200],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.01,
                                         n_epochs_rbm=5,
                                         n_iter_backprop=20,
                                         batch_size=10,
                                         activation_function='relu',
                                         dropout_p=10)
classifier.fit(X_train, y_train)
print(classifier.fit(X_train, y_train))
# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(y_test, Y_pred))

#minmaxed=encoded.copy()
#minmaxed_test = encoded_test.copy()
#
#a = list(minmaxed.columns.values)
#b = list(minmaxed_test.columns.values)
#
#c = list(set(a) - set(b))
#
#print(c)
#
##minmaxed[['spkts']] = scaler.fit_transform(minmaxed[['spkts']])
#mimmaxed = scaler.fit_transform(minmaxed)
#minmaxed_test = scaler.transform(minmaxed_test)
#
#
#for column in minmaxed:
#    minmaxed[[column]] = scaler.fit_transform(minmaxed[[column]])
#    
#for column in minmaxed_test:
#    minmaxed_test[[column]] = scaler.transform(minmaxed_test[[column]])
#

#new_Train_labels = Train_datasets.loc[:,'attack_cat']
#new_Test_labels = Test_datasets.loc[:,'attack_cat']
#convert_Train_labels = new_Train_labels.copy()
#convert_Test_labels = new_Test_labels.copy()
#
#attack_labels =new_Train_labels.unique()
#print(attack_labels)
#pos = np.where(attack_labels == 'Worms')
#pos = pos[0]
#print(pos)
#attack_labels.index('Normal')
#
##Convert Attack LAbels to numbers for Train Labels, take the index of unique value in the list
## of attacks and convert the name of attack to the position hold by the index.
#j = 0
#i = 0
#for j in range (len(new_Train_labels)):
#    for i in range (len(attack_labels)):
#        if attack_labels[i] == new_Train_labels[j]:
#            convert_Train_labels[j] =i
#            print (convert_Train_labels[j])
#
##Convert Attack Labels to unique numbers for Test Labels
#j = 0
#i = 0
#for j in range (len(new_Test_labels)):
#    for i in range (len(attack_labels)):
#        if attack_labels[i] == new_Test_labels[j]:
#            convert_Test_labels[j] =i
#            print (convert_Test_labels[j])
#            
#            
#            
#
#
##Do same thing for Training protocols
#protocol_unique = Train_datasets.loc[:,'proto'].unique()
#
#protocol_dict = {'tcp': 1,'udp': 2,'arp': 2, 'ospf': 2, 'icmp': 2, 'igmp': 2, 'rtp': 2, 'ddp': 2, 'ipv6-frag': 2, 'cftp': 2,
# 'wsn': 2, 'pvp': 2, 'wb-expak': 2, 'mtp': 2, 'pri-enc' : 2,'sat-mon': 2, 'cphb': 2, 'sun-nd' : 2,'iso-ip': 2,
# 'xtp': 2, 'il': 2, 'unas': 2, 'mfe-nsp' : 2,'3pc': 2, 'ipv6-route': 2, 'idrp': 2, 'bna' : 2,'swipe': 2,
# 'kryptolan': 2, 'cpnx': 2, 'rsvp': 2, 'wb-mon': 2, 'vmtp': 2, 'ib': 2, 'dgp': 2, 'eigrp': 2, 'ax.25': 2,
# 'gmtp': 2, 'pnni' : 2,'sep': 2, 'pgm' : 2,'idpr-cmtp' : 2,'zero' : 2,'rvd': 2, 'mobile': 2, 'narp': 2, 'fc': 2,
# 'pipe': 2, 'ipcomp': 2, 'ipv6-no' : 2,'sat-expak': 2, 'ipv6-opts': 2, 'snp' : 2,'ipcv': 2,
# 'br-sat-mon': 2, 'ttp': 2, 'tcf': 2, 'nsfnet-igp': 2, 'sprite-rpc' : 2,'aes-sp3-d': 2, 'sccopmce': 2,
# 'sctp' : 2,'qnx': 2, 'scps' : 2,'etherip': 2, 'aris' : 2,'pim' : 2,'compaq-peer': 2, 'vrrp': 2,'iatp': 2,
# 'stp' : 2,'l2tp': 2, 'srp' : 2,'sm' : 2,'isis': 2, 'smp' : 2,'fire' : 2,'ptp': 2, 'crtp': 2, 'sps': 2,
# 'merit-inp' : 2,'idpr': 2, 'skip': 2, 'any': 2, 'larp': 2, 'ipip': 2, 'micp': 2, 'encap': 2, 'ifmp': 2,
# 'tp++' : 2,'a/n' : 2,'ipv6': 2, 'i-nlsp' : 2,'ipx-n-ip' : 2,'sdrp' : 2,'tlsp': 2, 'gre' : 2,'mhrp': 2, 'ddx': 2,
# 'ippc' : 2,'visa': 2, 'secure-vmtp': 2, 'uti': 2, 'vines' : 2,'crudp': 2, 'iplt': 2, 'ggp' : 2,'ip': 2,
# 'ipnip' : 2,'st2': 2, 'argus': 2, 'bbn-rcc': 2, 'egp': 2, 'emcon': 2, 'igp': 2, 'nvp': 2, 'pup': 2, 'xnet': 2,
# 'chaos': 2, 'mux': 2, 'dcn': 2, 'hmp': 2, 'prm': 2, 'trunk-1': 2, 'xns-idp' : 2,'leaf-1': 2, 'leaf-2': 2,
# 'rdp' : 2,'irtp' : 2,'iso-tp4': 2, 'netblt': 2, 'trunk-2': 2, 'cbt': 2}
#i = 1
#for key,value in protocol_dict.items():
#    protocol_dict.update({key:i})
#    value = i
#    print (key,value)
#    i = i+1
#
##Replace protocols with unique array indexes
#Train_datasets.proto = [protocol_dict[item] for item in Train_datasets.proto] 
#print(Train_datasets.proto)
#
#Test_datasets.proto = [protocol_dict[item] for item in Test_datasets.proto]
#print(Test_datasets.proto)
##Replace service with uniue array indexes
#service_dict = {'-':1, 'ftp':2, 'smtp':3, 'snmp':4, 'http':5, 'ftp-data':6, 'dns':7, 'ssh':8,'radius':9, 'pop3':10,'dhcp':11, 'ssl':12, 'irc':13}
#Train_datasets.service =[service_dict[item] for item in Train_datasets.service]
#Test_datasets.service =[service_dict[item] for item in Test_datasets.service]
#
#state_unique = Train_datasets.loc[:,'state'].unique()
#state_unique_test = Test_datasets.loc[:,'state'].unique()
#print(state_unique)
#print(state_unique_test)
#state_dict = {'FIN':1, 'INT':2, 'CON':3, 'ECO':4, 'REQ':5, 'RST':6,'PAR':7, 'URN':8, 'no':9, 'ACC':10, 'CLO':11}
#Train_datasets.state =[state_dict[item] for item in Train_datasets.state]
#Test_datasets.state =[state_dict[item] for item in Test_datasets.state]
#
##Replace attack labels with unique array indexes
#Train_attack_label = Train_datasets.loc[:,'attack_cat'].unique()
#attack_dict = {'Normal':1, 'Backdoor':2, 'Analysis':3, 'Fuzzers':4, 'Shellcode':5, 'Reconnaissance':6,'Exploits':7, 'DoS':8, 'Worms':9,'Generic':10}
#Train_datasets.attack_cat =[attack_dict[item] for item in Train_datasets.attack_cat]
#Test_datasets.attack_cat = [attack_dict[item] for item in Test_datasets.attack_cat]
##
#
#new_Train_datasets = Train_datasets.drop(['id','attack_cat','label'], axis = 1)
#new_Test_datasets = Test_datasets.attack_cat


 







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

