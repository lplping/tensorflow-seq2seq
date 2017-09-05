#coding:utf-8
import numpy as np
import jieba
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
from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length
def get_encoder_layer(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):

    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, 
                                                      sequence_length=source_sequence_length, dtype=tf.float32)
    
    return encoder_output, encoder_state
def process_decoder_input(data, vocab_to_int, batch_size):

    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input
def decoding_layer(target_to_int, decoding_embedding_size, num_layers, rnn_size,target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
        
    
    
    target_vocab_size = len(target_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])
     
    # 
    output_layer = Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))


    
    with tf.variable_scope("decode"):
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer) 
        training_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
     
    with tf.variable_scope("decode", reuse=True):
        
        start_tokens = tf.tile(tf.constant([target_to_int['<GO>']], dtype=tf.int32), [batch_size], 
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                start_tokens,
                                                                target_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                        predicting_helper,
                                                        encoder_state,
                                                        output_layer)
        predicting_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_target_sequence_length)
    
    return training_decoder_output, predicting_decoder_output
def seq2seq_model(input_data, targets, lr, target_sequence_length, 
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size, 
                  rnn_size, num_layers):
    
    #
    _, encoder_state = get_encoder_layer(input_data, 
                                  rnn_size, 
                                  num_layers, 
                                  source_sequence_length,
                                  source_vocab_size, 
                                  encoding_embedding_size)
    
    
  
    decoder_input = process_decoder_input(targets, target_to_int, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(target_to_int, 
                                                                       decoding_embedding_size, 
                                                                       num_layers, 
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state, 
                                                                       decoder_input) 
    
    return training_decoder_output, predicting_decoder_output
    
epochs = 60

batch_size = 128

rnn_size = 128

num_layers = 2

encoding_embedding_size = 128
decoding_embedding_size = 128

learning_rate = 0.001

train_graph = tf.Graph()

with train_graph.as_default():
    
    # 获得模型输入    
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
    
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data, 
                                                                      targets, 
                                                                      lr, 
                                                                      target_sequence_length, 
                                                                      max_target_sequence_length, 
                                                                      source_sequence_length,
                                                                      len(source_to_int),
                                                                      len(target_to_int),
                                                                      encoding_embedding_size, 
                                                                      decoding_embedding_size, 
                                                                      rnn_size, 
                                                                      num_layers)    
    
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    
    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        
        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))
        
        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))
        
        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths
                                                                       
                                                                       
# 将数据集分割为train和validation
split_data=int(len(source_int) *0.2 )                                                                             
train_source = source_int[split_data:]
train_target = target_int[split_data:]
# 留出一个batch进行验证
valid_source = source_int[:split_data]
valid_target = target_int[:split_data]
print('####################################')   
print('trainq_length',len(train_source))  
print('traina_length',len(train_target)) 
print('valid_q_length',len(valid_source))  
print('valid_length',len(valid_target))
                                                                      
                                                                         
#==================================================================                                                                     
                                                                       
                                                                       
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, 
 valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                           source_to_int['<PAD>'],
                           target_to_int['<PAD>']))

display_step = 50 # 每隔50轮输出loss
                                                                                                                                      
#============================
current_epoch=0                                                                       
                                                                       
best_validation_loss=20                                                                       
checkpoint = "trained_model.ckpt" 
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
        
    for epoch_i in range(1, epochs+1):
                                                                     
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                           source_to_int['<PAD>'],
                           target_to_int['<PAD>'])):
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths})

            if batch_i % display_step == 0:
                
                # 计算validation loss
                validation_loss = sess.run(
                [cost],
                {input_data: valid_sources_batch,
                 targets: valid_targets_batch,
                 lr: learning_rate,
                 target_sequence_length: valid_targets_lengths,
                 source_sequence_length: valid_sources_lengths})
                
                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(train_source) // batch_size, 
                              loss, 
                              validation_loss[0]))
       
    saver = tf.train.Saver() 
    saver.save(sess, checkpoint)
    print('Model Saved')
                    