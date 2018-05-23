# coding:utf-8
# author: lichuang
# mail: shareditor.com@gmail.com
import sys
import numpy as np
import re
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import word_token
import jieba
import random

WHERE_RESTORE = True
# 输入序列长度
input_seq_len = 5
# 输出序列长度
output_seq_len = 5
# 空值填充0
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1
# 结尾标记
EOS_ID = 2
# LSTM神经元size
size = 8
# 初始学习率
init_learning_rate = 1
# 在样本中出现频率超过这个值才会进入词表
min_freq = 10

wordToken = word_token.WordToken()

# 放在全局的位置，为了动态算出num_encoder_symbols和num_decoder_symbols
max_token_id = wordToken.load_file_list(['/data/chatbot/ChatBotCourse-master/chatbotv1/samples/question', '/data/chatbot/ChatBotCourse-master/chatbotv1/samples/answer'], min_freq)
num_encoder_symbols = max_token_id + 5
num_decoder_symbols = max_token_id + 5


def get_id_list_from(sentence):
    sentence_id_list = []
    seg_list = jieba.cut(sentence)
    for str in seg_list:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(wordToken.word2id(str))
    return sentence_id_list

def seq_to_encoder(input_seq):
    """从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
    """
    input_seq_array = [int(v) for v in input_seq.split()]
    encoder_input = [PAD_ID] * (input_seq_len - len(input_seq_array)) + input_seq_array
    decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)
    encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
    decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
    target_weights = [np.array([1.0], dtype=np.float32)] * output_seq_len
    return encoder_inputs, decoder_inputs, target_weights


def get_model(feed_previous=False):
    """构造模型
    """

    learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in xrange(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in xrange(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in xrange(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in xrange(output_seq_len)]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=feed_previous,
                        dtype=tf.float32)

    # 计算加权交叉熵损失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    # 梯度下降优化器
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # 优化目标：让loss最小化
    update = opt.apply_gradients(opt.compute_gradients(loss))
    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate

if __name__ == "__main__":
    """
    预测过程
    """
    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model(feed_previous=True)
        saver.restore(sess, '/data/chatbot/ChatBotCourse-master/chatbotv1/model/demo')
#        sys.stdout.write("> ")
        sys.stdout.flush()
#        input_seq = sys.stdin.readline()
        input_seq = sys.argv[1]
#        while input_seq:
        if input_seq:
            input_seq = input_seq.strip()
            input_id_list = get_id_list_from(input_seq)
            if (len(input_id_list)):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(' '.join([str(v) for v in input_id_list]))

                input_feed = {}
                for l in xrange(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in xrange(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

                # 预测输出
                outputs_seq = sess.run(outputs, input_feed)
#                for logit in outputs_seq:
#                    print logit[0].shape
                # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
                outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
#                # 如果是结尾符，那么后面的语句就不输出了
                if EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
                outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                print re.sub('\s', '', " ".join(outputs_seq))
            else:
                print "WARN：词汇不在服务区"

#            sys.stdout.write("> ")
#            sys.stdout.flush()
#            input_seq = sys.stdin.readline()


