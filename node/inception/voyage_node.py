from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import cv2
import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception import inception_eval
from inception.voyage_data import VoyageData

FLAGS = tf.app.flags.FLAGS

class TrafficLightNode(object):
  def __init__(self):
    self.clf = None
    self.bridge = cv_bridge.CvBridge()
    self.label_pub = rospy.Publisher(
      'active_traffic_light', String, queue_size=10)
    self.img_sub = rospy.Subscriber(
      '/image_raw', Image, self.img_callback)

  def img_callback(self, msg):
    if not self.clf:
      self.clf = TrafficLightClassifier()

    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    label = self.clf.classify(img)
    self.label_pub.publish(label)


class TrafficLightClassifier(object):
  LABEL_NAMES = {
    0: 'unknown',
    1: 'none',
    2: 'green',
    3: 'red',
    4: 'red',
  }

  def __init__(self, input_shape=(600, 800, 3)):
    with tf.Graph().as_default():
      self.sess = tf.Session()
      size = FLAGS.image_size
      self.img_input = tf.placeholder(tf.uint8, shape=input_shape)
      image = tf.image.convert_image_dtype(self.img_input, dtype=tf.float32)
      image = tf.image.central_crop(image, central_fraction=0.875)
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [size, size],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)

      images = tf.reshape(image, (1, size, size, 3))
      labels = tf.placeholder(tf.int32, shape=(1, ))

      self.logits, _ = inception.inference(images, 5)

      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(self.sess, os.path.join(FLAGS.checkpoint_dir,
                                                ckpt.model_checkpoint_path))

      self.sess.graph.finalize()

  def classify(self, img):
    logits = self.sess.run([self.logits], feed_dict={self.img_input: img})
    p_label = np.argmax(logits)
    return self.LABEL_NAMES[p_label]

def accuracy():
  clf = TrafficLightClassifier(input_shape=(300, 400, 3))
  dataset_dir = '/fast/datasets/voyage/full_labeled'
  labels = np.loadtxt(os.path.join(dataset_dir, 'labels.csv'), delimiter=',')
  correct = 0.0
  start = time.time()
  predictions = {}
  actual = {}
  for timestamp, label in labels:
    img = np.load(os.path.join(dataset_dir, 'img', str(int(timestamp)) + '.npy'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label_name = clf.classify(img)
    predictions[label_name] = predictions.get(label_name, 0) + 1
    actual[label] = actual.get(label, 0) + 1
    if label_name == 'none' and label == 1.0:
      correct += 1
    elif label_name == 'green' and label == 2.0:
      correct += 1
    elif label_name == 'red' and label == 4.0:
      correct += 1

  print('accuracy', correct/len(labels))
  print('fps', len(labels) / (time.time() - start))
  print('predictions', predictions)
  print('actual', actual)

if __name__ == '__main__':
  if True:
    rospy.init_node('traffic_light_classifier')
    node = TrafficLightNode()
    rospy.spin()
  else:
    # evaluate accuracy on whole training data and record fps
    accuracy()
