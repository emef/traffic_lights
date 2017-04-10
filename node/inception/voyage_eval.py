from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception import inception_eval
from inception.voyage_data import VoyageData

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
  dataset = VoyageData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  inception_eval.evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
