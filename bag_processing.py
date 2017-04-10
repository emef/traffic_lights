import errno
import os
import shutil
import sys
import termios
import tty

import cv2
import cv_bridge
import numpy as np
import rosbag
from Tkinter import *
from PIL import ImageTk, Image

CAMERA_TOPIC = '/image_raw'


class BagExtractor(object):
    def __init__(self,
                 bagfile,
                 callback,
                 start=None,
                 stop=None,
                 interval=1):
        self.bagfile = bagfile
        self.callback = callback
        self.start = start
        self.stop = stop
        self.interval = interval

    def run(self):
        bag = rosbag.Bag(self.bagfile)
        first_secs, latest_interval = None, None
        for _, msg, t in bag.read_messages(CAMERA_TOPIC):
            t_millis = int(t.to_sec() * 1000)
            t_secs = int(t.to_sec())
            if not first_secs:
                first_secs = t_secs

            offset_secs = t_secs - first_secs
            if self.start and offset_secs < self.start:
                continue
            if self.stop and offset_secs > self.stop:
                break

            t_interval = (
                None if self.interval is None
                else t_secs / self.interval)

            if not self.interval or t_interval > latest_interval:
                self.callback(t_millis, msg)

            latest_interval = t_interval

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def extract_to_path(
        bagfile='/fast/datasets/voyage/clipped-mtv-sf-1.orig.bag',
        output_dir='/fast/datasets/voyage/1sec/'):
    bridge = cv_bridge.CvBridge()
    mkdir_p(os.path.dirname(output_dir))

    def callback(t, msg):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        png_path = os.path.join(output_dir, str(t) + '.png')
        cv2.imwrite(png_path, img)

    BagExtractor(bagfile, callback).run()

def label_images(
        input_dir='/fast/datasets/voyage/1sec/img',
        output_file='/fast/datasets/voyage/1sec/labels.csv'):
    tk = Tk()
    panel = Label(tk)

    filenames = list(os.listdir(input_dir))
    filenames.sort()
    timestamps = [int(filename.split('.')[0]) for filename in filenames]
    labels = [None for _ in xrange(len(filenames))]
    i = 0
    done = False

    while not done and i < len(filenames):
        filename = filenames[i]
        img = cv2.imread(os.path.join(input_dir, filename))
        tk_img = ImageTk.PhotoImage(Image.fromarray(img))
        panel.config(image=tk_img)
        panel.pack(side='top', fill='both', expand='yes')
        tk.update()

        print '1 = NONE | 2 = GREEN | 3 = yellow | 4 = red'

        while True:
            c = get_ch()
            if c == '\x1b':
                get_ch()
                get_ch()
                i = max(i-1, 0)
                break
            elif c in ('1', '2', '3', '4'):
                labels[i] = int(c)
                i += 1
                break
            elif ord(c) < 10:
                done = True
                break

    if os.path.exists(output_file):
        backup_file = output_file + '.bak'
        print 'backed up old labels to', backup_file
        shutil.move(output_file, backup_file)

    labeled = i
    with open(output_file, 'w') as f:
        for i in xrange(labeled):
            f.write('%d,%d\n' % (timestamps[i], labels[i]))

def finalize_dataset(
        bagfile='/fast/datasets/voyage/clipped-mtv-sf-1.orig.bag',
        labelfile='/fast/datasets/voyage/1sec/labels.csv',
        output_dir='/fast/datasets/voyage/full_labeled'):

    bridge = cv_bridge.CvBridge()
    mkdir_p(output_dir)
    mkdir_p(os.path.join(output_dir, 'img'))

    ref_labels = np.loadtxt(labelfile, delimiter=',', dtype='int')
    labels = []

    def callback(t, msg):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        npy_path = os.path.join(output_dir, 'img', str(t) + '.npy')
        np.save(npy_path, img)

        ix = np.searchsorted(ref_labels[:, 0], t, side='left')
        ix = max(min(ix, len(ref_labels) - 1), 0)
        labels.append((t, ref_labels[ix, 1]))

    BagExtractor(bagfile, callback, interval=None).run()

    output_labelfile = os.path.join(output_dir, 'labels.csv')
    with open(output_labelfile, 'w') as f:
        for t, label in labels:
            f.write('%d,%d\n' % (t, label))

LABELS = {
    1: 'none',
    2: 'green',
    3: 'red', # don't support yellow lights
    4: 'red',
}

def prepare_tensorflow(
        dataset_dir='/fast/datasets/voyage/full_labeled',
        output_dir='/fast/datasets/voyage/tfr_convertible',
        train_pct=0.7,
        shuffle=False):

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    mkdir_p(train_dir)
    mkdir_p(test_dir)

    for label_name in set(LABELS.values()):
        mkdir_p(os.path.join(train_dir, label_name))
        mkdir_p(os.path.join(test_dir, label_name))

    labels = np.loadtxt(os.path.join(dataset_dir, 'labels.csv'), delimiter=',')

    if shuffle:
        np.random.shuffle(labels)

    n_train = int(train_pct * len(labels))
    train_labels = labels[:n_train]
    test_labels = labels[n_train:]

    for t, label in train_labels:
        img_src = os.path.join(dataset_dir, 'img', '%d.npy' % t)
        img_dest = os.path.join(train_dir, LABELS[label], '%d.png' % t)
        img_dest_flip = os.path.join(train_dir, LABELS[label], '%d-flip.png' % t)
        npy_to_png(img_src, img_dest, False)
        npy_to_png(img_src, img_dest_flip, True)

    for t, label in test_labels:
        img_src = os.path.join(dataset_dir, 'img', '%d.npy' % t)
        img_dest = os.path.join(test_dir, LABELS[label], '%d.png' % t)
        img_dest_flip = os.path.join(test_dir, LABELS[label], '%d-flip.png' % t)
        npy_to_png(img_src, img_dest, False)
        npy_to_png(img_src, img_dest_flip, True)

    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        for label_name in set(LABELS.values()):
            f.write(label_name + '\n')

def npy_to_png(img_src, img_dest, flip):
    img = np.load(img_src)
    if flip:
        img = img[:, ::-1, :]
    cv2.imwrite(img_dest, img)


def get_ch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


COMMANDS = {
    'extract': extract_to_path,
    'label': label_images,
    'finalize': finalize_dataset,
    'tensorflow': prepare_tensorflow,
}

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) else None
    if not cmd or cmd not in COMMANDS:
        print 'Usage: ' ' | '.join(COMMANDS)


    COMMANDS[cmd]()
