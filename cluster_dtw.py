from pydub import AudioSegment
import json
import numpy as np
import python_speech_features
import pydub
import matplotlib.pyplot as plt
from pydub import AudioSegment
import IPython
import scipy
from python_speech_features import logfbank
from python_speech_features import mfcc
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pydub
import scipy.spatial.distance as ssd
from scipy.spatial.distance import pdist, euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
import os
import math
from collections import Counter
import time
from sklearn.metrics import davies_bouldin_score, silhouette_score

# choose the label and phone

label_choice = input("Choose Label: accent (a) | phone (p) | recording (r): ")
phone_choice = input("Select phone: all (a) | distinguished (d) | [type phone]: ")

all = True
phone_list = []
# deciding the phone list
if phone_choice != "a":
    all = False
    if phone_choice == 'd':
        phone_list = ['ow', 'ih', 'er', 'aa']
    else:
        phone_list = [phone_choice]


#phones_all = []


# setting the folder
audio_samples = os.path.join(os.getcwd(), "females")

cluster_str = "cluster_" + str(phone_choice)
label_str = "label_" + str(label_choice)

cluster_dir = os.path.join(os.getcwd(), cluster_str)
label_dir = os.path.join(os.getcwd(), label_str)
os.mkdir(cluster_dir)
os.mkdir(label_dir)
accents = {}


for audio_name in os.listdir(audio_samples):
    if ".DS_Store" in audio_name:
        continue
    if not all:
        if audio_name[:2] in phone_list:

            # audio to mfccs
            audio = os.path.join("females", audio_name)
            #print(audio)
            segment = AudioSegment.from_file(audio)
            samples = segment.get_array_of_samples()
            fbank_feat = mfcc(np.array(samples))
            m, n = fbank_feat.shape
            fbank_feat = fbank_feat.reshape(1, m*n)
            # all_mfccs.append(fbank_feat)

            name_split = audio_name.split("_")
            # extract label
            if label_choice == 'a':
                label = name_split[2]
                #labels.append(label)


            elif label_choice == 'p':
                label = name_split[0] + "_" + name_split[1]
                #labels.append(label)



            else:
                label = ""
                for word in name_split[4:8]:
                    label += word
                #labels.append(label)

            if name_split[2] in accents:
                accents[name_split[2]]['audio'].append(fbank_feat)
                accents[name_split[2]]['label'].append(label)
            else:
                accents[name_split[2]] = {
                    'audio': [fbank_feat],
                    'label': [label]
                }


    else:

        # audio to mfccs
        audio = os.path.join("females", audio_name)
        # print(audio)
        segment = AudioSegment.from_file(audio)
        samples = segment.get_array_of_samples()
        fbank_feat = mfcc(np.array(samples))
        m, n = fbank_feat.shape
        fbank_feat = fbank_feat.reshape(1, m*n)
        #all_mfccs.append(fbank_feat)

        name_split = audio_name.split("_")
        # extract label
        if label_choice == 'a':
            label = name_split[2]
            #labels.append(label)


        elif label_choice == 'p':
            label = name_split[0] + "_" + name_split[1]
            #labels.append(label)



        else:
            label = ""
            for word in name_split[4:8]:
                label += word
            #labels.append(label)


        if name_split[2] in accents:
            accents[name_split[2]]['audio'].append(fbank_feat)
            accents[name_split[2]]['label'].append(label)
        else:
            accents[name_split[2]] = {
                'audio': [fbank_feat],
                'label': [label]
            }

# print(phones_all)

'''
m = ([len(accents[i]) for i in accents])
m = np.min(m)
new_list=[]
for i in accents:
    s = accents[i]
    s = s[:m]
    accents[i] = s
    new_list = new_list + s

print(new_list)
new=[]
# similarity metric to compare all recordings to each other
for i in range(len(all_mfccs)):
    if i in new_list:
        new.append(all_mfccs[i])
all_mfccs = new



print(all_mfccs)




#mfccs_new = np.asarray(new)



# similarity metric to compare all recordings to each other


x = len(all_mfccs)

similarity = np.zeros((x,x))

for i in range(x):
    m1 = all_mfccs[i]
    for j in range(x):
        m2 = all_mfccs[j]
        similarity[i,j],_ = fastdtw(m1.T,m2.T, dist=euclidean)
'''

all_mfccs = []
labels = []

shortest_length = math.inf
for data in accents.values():
    if len(data['audio']) < shortest_length:
        shortest_length = len(data['audio'])

print(shortest_length)



for data in accents.values():
    for aud in data['audio'][:shortest_length]:
        all_mfccs.append(aud)
    #print(data['label'])
    for label in data['label'][:shortest_length]:
        labels.append(label)



#print(labels)
#print(all_mfccs)

ds = dtw.distance_matrix_fast(all_mfccs)

ds[ds == math.inf] = 0
ds_T = ds.T
ds = ds+ds_T

dist_mat = ssd.squareform(ds, checks=False)



#print(ds)
#print()


# dendrogram to cluster



linked = linkage(dist_mat, 'ward')



sch.dendrogram(linked)
plt.gcf()
plt.savefig("dendrogram_"+str(phone_choice)+"_sample.png")
plt.show()




#time.sleep(5)

# now actually choosing the cluster
num_clusters = input("How many clusters? ")

clustering = AgglomerativeClustering(n_clusters = int(num_clusters), linkage='ward').fit(ds)
#print(clustering.labels_)
#print(len(clustering.labels_))
assigned_clusters = np.array(clustering.labels_)

#print(assigned_clusters)

#print(Counter(labels))

accent_dict = {}

# for each cluster
for cluster in range(int(num_clusters)):
    plt.figure()
    within = np.where(assigned_clusters == cluster)[0]

    print(cluster, len(within))
    print()
    cluster_dict = {}

    for index in within:
        if labels[index] in cluster_dict:
            cluster_dict[labels[index]] += 1
        elif not labels[index] in cluster_dict:
            cluster_dict[labels[index]] = 1
        if labels[index] in accent_dict:
            accent_dict[labels[index]].append(cluster)
        elif labels[index] not in accent_dict:
            accent_dict[labels[index]] = [cluster]



    vals = list(cluster_dict.values())
    keys = list(cluster_dict.keys())

    max = np.argmax(vals)
    '''
    print(vals)
    print(max)
    print(keys[max])
    '''

    exp = np.zeros(len(vals))
    exp[max] = 0.1

    # now create a chart
    plt.pie(vals, explode = exp, autopct='%1.0f%%', labels=keys, shadow=True, startangle=90)

    plt.gcf()
    plt.savefig(cluster_str + "/cluster_" + str(cluster) + ".png")

#print(accent_dict)

print('db is',davies_bouldin_score(ds, assigned_clusters))
print()
print('silhouette_score', silhouette_score(ds, assigned_clusters, metric="precomputed"))
print()

for key,vals in accent_dict.items():
    # value is the vector of labels
    plt.figure()
    counts = Counter(vals)
    labels = list(counts.keys())
    #print(labels)
    values = list(counts.values())
    #print(values)

    max = np.argmax(values)
    exp = np.zeros(len(values))
    exp[max] = 0.1

    plt.pie(values, explode = exp, autopct='%1.0f%%', labels=labels, shadow=True, startangle=90)

    plt.gcf()
    plt.savefig(label_str + "/" + str(key)+".png")



print("All done!")
