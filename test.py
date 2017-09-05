#coding:utf-8
import numpy as np
import jieba
import tensorflow as tf
batch_size=128
def load_data(path):
    data=[]
    with open(path,'r') as f:
        for line in f:
            data.append(line)
    return data

def cret_dict(data):
    codes=['<PAD>','<UNK>','<GO>','<EOS>']
    set_words=set([term for line in data for term in line.split()])
    int_to_vab={word_i:word for word_i,word in enumerate(codes+list(set_words))}
    vab_to_int={word:word_i for word_i,word in int_to_vab.items()}
    return int_to_vab,vab_to_int

source='./data/trainenc.txt'
target_path = './data/traindec.txt'
q=load_data(source)
a=load_data(target_path)
q_seg=[' '.join(jieba.cut(line)).encode('utf-8').strip() for line in q]
a_seg=[' '.join(jieba.cut(line)).encode('utf-8').strip() for line in a]
#=======================
int_to_source,source_to_int=cret_dict(q_seg)
int_to_target,target_to_int=cret_dict(a_seg)
#对字母进行转化

source_int=[[source_to_int.get(term,source_to_int['<UNK>']) 
             for term in line .split() ] for line in q_seg]
target_int=[[target_to_int.get(term,target_to_int['<UNK>']) for term in line.split()] + [target_to_int['<EOS>']] for line in a_seg]
print(source_int[:3])

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 40
    return [source_to_int.get(word, source_to_int['<UNK>']) for word in text.split()] + [source_to_int['<PAD>']]*(sequence_length-len(text))
	# 输入一个单词

input_word = '您要是真不要命那今儿您就唱 '
input_word=' '.join(jieba.cut(input_word)).encode('utf-8')
print(input_word)                                                  
text = source_to_seq(input_word)

checkpoint = "./trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    
    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      target_sequence_length: [len(input_word)]*batch_size, 
                                      source_sequence_length: [len(input_word)]*batch_size})[0] 


pad = source_to_int["<PAD>"] 

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(' '.join([int_to_source[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_target[i] for i in answer_logits if i != pad])))