import numpy as np
import cv2
import os
import tensorflow as tf
import random
import dlib

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class facial_expression():
    def __init__(self):

        self.landmark_data = []
        self.X = tf.placeholder(tf.float32, [None, 2304])
        self.Y = tf.placeholder(tf.int32, [None])  # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

        self.landmark_X = tf.placeholder(tf.float32, [None, 144])

        self.keep_prob = tf.placeholder(tf.float32)
        self.checkpoint_save_dir = os.path.join("/home/hci/hyeon/git/FEM2/hia123")
        self.data_file_path = os.path.join("data_set", "fer2013.csv")

        # self.loss, self.decoded = self.autoencoder(self.X)
        # self.train_step = tf.train.AdagradOptimizer(0.1).minimize(self.loss)

        self.X_img = tf.reshape(self.X, [-1, 48, 48, 1])
        self.landmark_img = tf.reshape(self.landmark_X, [-1, 12, 12, 1])
        self.Y_one_hot = tf.one_hot(self.Y, 7)

        # 1st Layer
        self.weight_1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        self.bias_1 = tf.Variable(tf.random_normal([32], stddev=0.01))

        self.L1 = tf.nn.conv2d(self.X_img, self.weight_1, strides=[1, 1, 1, 1], padding='SAME')
        self.L1 = tf.nn.bias_add(self.L1, self.bias_1)

        self.L1 = tf.nn.relu(self.L1)
        self.L1 = tf.nn.max_pool(self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME')  # now size becself.data = {}omes ? 24 24 32
        # landmark 1st Layer
        self.land_L1 = tf.nn.conv2d(self.landmark_img, self.weight_1, strides=[1, 1, 1, 1], padding='SAME')
        self.land_L1 = tf.nn.bias_add(self.land_L1, self.bias_1)

        self.land_L1 = tf.nn.relu(self.land_L1)
        self.land_L1 = tf.nn.max_pool(self.land_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')

        # self.L1 = tf.nn.dropout(self.L1, keep_prob=self.keep_prob)

        # 2nd Layer
        self.weight_2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        self.bias_2 = tf.Variable(tf.random_normal([64], stddev=0.01))

        self.L2 = tf.nn.conv2d(self.L1, self.weight_2, strides=[1, 1, 1, 1], padding='SAME')
        self.L2 = tf.nn.bias_add(self.L2, self.bias_2)

        self.L2 = tf.nn.relu(self.L2)
        self.L2 = tf.nn.max_pool(self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME')  # now size becomes ? 12 12 64

        # landmark 2nd Layer
        self.land_L2 = tf.nn.conv2d(self.land_L1, self.weight_2, strides=[1, 1, 1, 1], padding='SAME')
        self.land_L2 = tf.nn.bias_add(self.land_L2, self.bias_2)

        self.land_L2 = tf.nn.relu(self.land_L2)
        self.land_L2 = tf.nn.max_pool(self.land_L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')  # now size becomes ? 12 12 64
        # self.L2 = tf.nn.dropout(self.L2, keep_prob=self.keep_prob)

        # 3rd Layer
        self.weight_3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        self.bias_3 = tf.Variable(tf.random_normal([128], stddev=0.01))

        self.L3 = tf.nn.conv2d(self.L2, self.weight_3, strides=[1, 1, 1, 1], padding='SAME')
        self.L3 = tf.nn.bias_add(self.L3, self.bias_3)
        self.L3 = tf.nn.relu(self.L3)
        self.L3 = tf.nn.max_pool(self.L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME')  # now size becomes ? 6 6 128

        # self.L3 = tf.nn.dropout(self.L3, keep_prob=self.keep_prob)
        self.L3_flatten = tf.reshape(self.L3, [-1, 6 * 6 * 128])
        self.landmark_flatten = tf.reshape(self.land_L2, [-1, 3 * 3 * 64])

        # FC Layer
        # self.weight_4 = tf.get_variable(name="w4", shape=[6 * 6 * 128, 7],
        #                                 initializer=tf.contrib.layers.xavier_initializer())

        self.weight_4 = tf.get_variable(name="w4", shape=[(6 * 6 * 128), 7],
                                        initializer=tf.contrib.layers.xavier_initializer())
        # self.landmark_weight_4 = tf.get_variable(name="w4", shape=[(3 * 3 * 128), 7],
        #                                 initializer=tf.contrib.layers.xavier_initializer())
        self.landmark_weight = tf.get_variable(name="w5", shape=[3 * 3 * 64, 7], initializer=tf.contrib.layers.xavier_initializer())
        self.bias_4 = tf.Variable(tf.random_normal([7]))
        self.logits = tf.add(tf.matmul(self.L3_flatten,self.weight_4),tf.matmul(self.landmark_flatten,self.landmark_weight)) + self.bias_4  # size now become ?,7
        # self.landmark_logidits = tf.matmul(self.landmark_flatten, self.landmark_weight) + self.bias_4
        self.softmax_logits = tf.nn.softmax(self.logits)

        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_one_hot, logits=self.logits)
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.prediction = tf.argmax(self.logits, 1)

        self.prediction_result = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y_one_hot, 1))

        self.mean_cost = tf.reduce_mean(self.cost, 0)
        self.mean_accuracy = tf.reduce_mean(tf.cast(self.prediction_result, tf.float32))


    def get_dataset(self, file_path, batch_size, num_fake_img=0, shuffle=True):
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

                    for j in range(num_fake_img):
                        choose_fake_img_type = random.randrange(0, 2)
                        x_rows = []
                        if choose_fake_img_type == 0:
                            noise = random.randrange(-8, 8) / 10

                            for image_data in txt_batch[1].split():
                                image_data = int(image_data)
                                if noise > 0:
                                    image_data = image_data + ((255 - image_data) * noise)
                                else:
                                    image_data = image_data + ((image_data) * noise)
                                x_rows.append(np.uint8(image_data))
                            images.append(x_rows)
                            labels.append(txt_batch[0])
                            train_or_test.append(txt_batch[2])

                        x_rows = []
                        noise_alpha = 0.1
                        if choose_fake_img_type == 1:
                            for image_data in txt_batch[1].split():
                                image_data = int(image_data)
                                noise = random.randrange(int(-1 * image_data * noise_alpha),
                                                         int((256 - image_data) * noise_alpha))
                                image_data += noise
                                x_rows.append(np.uint8(image_data))

                            images.append(x_rows)
                            labels.append(txt_batch[0])
                            train_or_test.append(txt_batch[2])

                        for j in range(len(images)):
                            fake_img_shuffle = random.randrange(0, len(images))
                            images[j], images[fake_img_shuffle] = images[fake_img_shuffle], images[j]
                            labels[j], labels[fake_img_shuffle] = labels[fake_img_shuffle], labels[j]
                            train_or_test[j], train_or_test[fake_img_shuffle] = train_or_test[fake_img_shuffle], \
                                                                                train_or_test[j]

                # yield images, labels, train_or_test
                yield images, labels, train_or_test

    def get_landmarks(self, image):
        detections = detector(image, 1)
        for k, d in enumerate(detections):
            shape = predictor(image, d)
            xlist = []
            ylist = []
            for i in range(0, 68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xcentral = [(x - xmean) for x in xlist]
            ycentral = [(y - ymean) for y in ylist]
            landmarks_vectorised = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)

            for i in range(0, 8):
                landmarks_vectorised.append(0)

            # self.data['landmarks_vectorised'] = landmarks_vectorised
            self.landmark_data.append(landmarks_vectorised)

        if len(detections) < 1:
            landmarks_vectorised_x = "error"
            return False

    def load_graph(self, sess, checkpoint_save_dir):
        saver = tf.train.Saver(max_to_keep=1000)
        latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_save_dir)  # old to new
        saver.restore(sess, latest_chkpt)

    def start_train(self, checkpoint_save_dir, epoch, batch_size, num_fake_img, file_path, eval_freq):
        if not os.path.exists(checkpoint_save_dir):
            print("check point dir not found, making one")
            os.mkdir(checkpoint_save_dir)
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
            delete_list = []
            y = []
            for i in range(epoch):
                print("%s epoch" % i)
                for image_data, label_data, train_or_test in self.get_dataset(file_path, batch_size, num_fake_img):
                    for x in range(batch_size):
                        pixels_array = np.asarray(image_data[x])
                        image = pixels_array.reshape(48, 48)
                        if self.get_landmarks(image) is False:
                            delete_list.append(x)

                    for y in reversed(delete_list):
                        del (image_data[y])
                        del (label_data[y])

                    step += 1
                    if (step % eval_freq == 0):
                        print("-------------------------------------------------------------------------------------")
                        # cost, accuracy = sess.run([self.mean_cost, self.mean_accuracy],
                        #                           feed_dict={self.X: image_data, self.Y: label_data, self.keep_prob: 1})
                        cost, accuracy = sess.run([self.mean_cost, self.mean_accuracy],
                                                  feed_dict={self.X: image_data, self.Y: label_data})
                        print("COST: ", cost)
                        print("Accuracy: ", accuracy)
                        saver.save(sess, os.path.join(checkpoint_save_dir, "facial expression"))
                        print("progress saved at %s " % checkpoint_save_dir + "%s epoch %s step" % (i, step))
                        print("-------------------------------------------------------------------------------------")
                        print(self.landmark_data)
                        self.landmark_data = []
                        delete_list = []
                    else:
                        # sess.run(self.train, feed_dict={self.X: image_data, self.Y: label_data, self.keep_prob: 0.7})
                        sess.run(self.train, feed_dict={self.X: image_data, self.Y: label_data})
                        self.landmark_data = []
                        delete_list = []


if __name__ == '__main__':
    # images, labels, train = get_dataset("/home/hci/PycharmProjects/hklovelovehs/fer2013.csv", 50)
    a = facial_expression()
    a.start_train(a.checkpoint_save_dir, 20, 50, 0, a.data_file_path, 10)