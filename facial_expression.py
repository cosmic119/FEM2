import numpy as np
import cv2
import os
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 2304])
Y = tf.placeholder(tf.int32, [None])  # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
keep_prob = tf.placeholder(tf.float32)
checkpoint_save_dir = "/home/hci/PycharmProjects/hklovelovehs/checkpoint"
data_file_path = "/home/hci/PycharmProjects/hklovelovehs/fer2013.csv"

mean_cost = 0
mean_accuracy = 0
train = 0


def cnn_model():
    checkpoint_save_dir = os.path.join("checkpoint")
    file_path = os.path.join("data_set", "fer2013.csv")

    X_img = tf.reshape(X, [-1, 48, 48, 1])
    Y_one_hot = tf.one_hot(Y, 7)

    # 1st Layer
    weight_1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    bias_1 = tf.Variable(tf.random_normal([32], stddev=0.01))

    L1 = tf.nn.conv2d(X_img, weight_1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.bias_add(L1, bias_1)

    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')  # now size becomes ? 24 24 32

    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # 2nd Layer
    weight_2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    bias_2 = tf.Variable(tf.random_normal([64], stddev=0.01))

    L2 = tf.nn.conv2d(L1, weight_2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.bias_add(L2, bias_2)

    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')  # now size becomes ? 12 12 64

    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    # 3rd Layer
    weight_3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    bias_3 = tf.Variable(tf.random_normal([128], stddev=0.01))

    L3 = tf.nn.conv2d(L2, weight_3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.bias_add(L3, bias_3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')  # now size becomes ? 6 6 128

    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flatten = tf.reshape(L3, [-1, 6 * 6 * 128])

    # FC Layer
    weight_4 = tf.get_variable(name="w4", shape=[6 * 6 * 128, 7], initializer=tf.contrib.layers.xavier_initializer())
    bias_4 = tf.Variable(tf.random_normal([7]))

    logits = tf.matmul(L3_flatten, weight_4) + bias_4  # size now become ?,7
    softmax_logits = tf.nn.softmax(logits)

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=logits)
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    prediction = tf.argmax(logits, 1)

    prediction_result = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))

    mean_cost = tf.reduce_mean(cost, 0)
    mean_accuracy = tf.reduce_mean(tf.cast(prediction_result, tf.float32))


def get_dataset(file_path, batch_size, shuffle=True):
    with open(file_path) as csvfile:
        print("file opening")
        csvfile = csvfile.readlines()[1:-31]
        csvfile = np.array(csvfile)

        if shuffle:
            np.random.shuffle(csvfile)

        # except int()
        for i in range(int(round(len(csvfile) / batch_size, 0))):
            labels = []
            images = []
            train_or_test = []

            txt_list = csvfile[i * batch_size:i * batch_size + batch_size]
            for txt_batch in txt_list:
                txt_batch = txt_batch.split(",")
                images.append([np.uint8(image_data) for image_data in txt_batch[1].split()])
                labels.append(txt_batch[0])
                train_or_test.append([txt_batch[2]])

            # yield images, labels, train_or_test
            yield images, labels, train_or_test


def start_train(checkpoint_save_dir, epoch, batch_size, file_path, eval_freq):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1000)
        latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_save_dir)  # old check change to new
        if latest_chkpt == None:
            print("no checkpoint was found")
        else:
            print("checkpoint found at: %s" % latest_chkpt)
            saver.restore(sess, latest_chkpt)

        step = 0
        for i in range(epoch):
            print("%s epoch" % i)
            for image_data, label_data, train_or_test in get_dataset(file_path, batch_size):
                if (step % eval_freq == 0):
                    print("-------------------------------------------------------------------------------------")
                    cost, accuracy = sess.run([mean_cost, mean_accuracy],
                                              feed_dict={X: image_data, Y: label_data, keep_prob: 1})
                    print("COST: ", cost)
                    print("Accuracy: ", accuracy)
                    saver.save(sess, os.path.join(checkpoint_save_dir, "facial expression"))
                    print("progress saved at %s " % checkpoint_save_dir + "%s epoch %s step" % (i, step))
                    print("-------------------------------------------------------------------------------------")
                else:
                    sess.run(train, feed_dict={X: image_data, Y: label_data, keep_prob: 0.7})


if __name__ == '__main__':
    # images, labels, train = get_dataset("/home/hci/PycharmProjects/hklovelovehs/fer2013.csv", 50)
    # print(images)
    # print(labels)
    # print(train)
    start_train(checkpoint_save_dir, 20, 50, data_file_path, 10)
