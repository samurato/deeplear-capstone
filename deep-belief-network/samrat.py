import os


import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class Dataset(object):
    def __init__(self, segments, labels, one_hot = False, dtype = dtypes.float32, reshape = True):
        """Construct a Dataset
        one_hot arg is used only if fake_data is True. 'dtype' can be either unit9 or float32
        """

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid')

        self._num_examples = segments.shape[0]
        self._segments = segments
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def segments(self):
        return self._segments

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next batch-size examples from this dataset"""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed +=1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._segments = self._segments[perm]
            self._labels = self._labels[perm]

            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._segments[start:end,:, :], self._labels[start:end,:]












def nominal(dataset1, dataset2):
    dataset = dataset1.append(dataset2)
    protocol1 = dataset1['proto'].copy()
    protocol2 = dataset2['proto'].copy()
    protocol_type = dataset['proto'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
        service2[service2 == service_type[i]] = i
    state1 = dataset1['state'].copy()
    state2 = dataset2['state'].copy()
    state = pd.concat([state1, state2])
    print(state)
    state_type = state.unique()
    for i in range(len(state_type)):
        state1[state1 == state_type[i]] = i
        state2[state2 == state_type[i]] = i
    return protocol1, service1, state1, protocol2, service2, state2

#Reads Files saves the col_name and 
def read_data(filename):
    col_names = ["id", "dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports","attack","label"]
    dataset = pd.read_csv(filename, header = 1, names = col_names, index_col = "id")
    return dataset  

def read_data_set(dataset1, dataset2, one_hot = False, dtype = dtypes.float32, reshape = True):

    segments1, labels1 = segment_signal(dataset1)
    #labels1 = np.asarray(pd.get_dummies(labels1), dtype = np.int8)

    segments2, labels2 = segment_signal(dataset2)
    #labels2 = np.asarray(pd.get_dummies(labels2), dtype = np.int8)
    labels = np.asarray(pd.get_dummies(labels1.append([labels2])), dtype = np.int8)
    labels1 = labels[:len(labels1)]
    labels2 = labels[len(labels1):]
    train_x = segments1.reshape(len(segments1), 1, 1 ,42)
    train_y = labels1

    test_x = segments2.reshape(len(segments2), 1, 1 ,42)
    test_y = labels2

    train = Dataset(train_x, train_y, dtype = dtype , reshape = reshape)
    test = Dataset(test_x, test_y, dtype = dtype, reshape = reshape)
    return base.Datasets(train = train, validation=None, test = test)   



def segment_signal(data, window_size = 1):
    segments = np.empty((0, window_size, 42))
    labels = np.empty((0))
    num_features = ["dur","proto","service", "state", "spkts","dpkts","sbytes","dbytes",
    "rate","sttl", "dttl","sload","dload","sloss","dloss",
    "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt",
    "synack","ackdat","smean","dmean","trans_depth","response_body_len",
    "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm",
    "ct_src_dst", "is_sm_ips_ports"
    ]
    segments = np.asarray(data[num_features].copy())
    labels = data["attack"]

    return segments, labels



dir_path = os.path.dirname(os.path.realpath(__file__))
filename2 = dir_path + "/datasets/unsw/UNSW_NB15_testing-set.csv"
filename1 = dir_path + "/datasets/unsw/UNSW_NB15_training-set.csv"

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
print(acc)
X_train = dataset2
X_test = labels2
Y_train = dataset1
Y_test = labels1

classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)