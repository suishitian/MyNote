import tensorflow as tf
import math
import os
import random
import zipfile
import numpy as np
import urllib
import collections

url = "http://mattmahoney.net/dc/"
def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception(
          "Failed to verify" + filename +". Con you get to it with a browser?"
        )
    return filename

filename = "text8.zip"

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)

print('Data size',len(words))

v_size = 50000

def build_dataset(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(v_size - 1))
    dictr = dict()
    for word,_ in count:
        dictr[word] = len(dictr)
    data = list()
    unk_count=0
    for word in words:
        if word in dictr:
            index = dictr[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    r_dict = dict(zip(dictr.values(),dictr.keys()))
    return data,count,dictr,r_dict
data,count,dict,r_dict = build_dataset(words)
##del words
print('Most common ,',count[:5])
print('Sample data,',data[:10],[r_dict[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size%num_skips==0
    assert num_skips <=2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)

    span = 2*skip_window+1
    buffer=collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch,labels

batch,labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i],r_dict[batch[i]],"-----",labels[i,0],r_dict[labels[i,0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skip = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window,valid_size,replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype = tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([v_size,embedding_size],-1.0,1.0))
        embed = tf.nn.embedding_lookup(embeddings,train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([v_size,embedding_size],stddev=1.0/math.sqrt(embedding_size))
        )
        nce_biases = tf.Variable(tf.zeros([v_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=v_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings,transpose_b=True)

    init = tf.global_variables_initializer()
num_steps = 100001
with tf.Session(graph=graph) as sess:
    init.run()
    print("init")

    ave_loss = 0
    for step in range(num_steps):
        batch_inputs,batch_labels = generate_batch(batch_size,num_skip,skip_window)

        feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}

        _,loss_val = sess.run([optimizer,loss],feed_dict=feed_dict)
        ave_loss += loss_val

        if step %200 ==0:
            if step>0:
                ave_loss /=200
            print("ave_loss: ",step,":",ave_loss)

        if step%2000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = r_dict[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                str = "Nearest to %s :" % valid_word
                for k in range(top_k):
                    close_word = r_dict[nearest[k]]
                    str = "%s %s," %(str,close_word)
                print(str)
    final_embeddings = normalized_embeddings.eval()