### bag_processing.py

There are four commands in this here:

`extract` - Extract all the images from the bag to pngs
`label` - A quick n' dirty tk gui app to rapidly label the images at 1s intervals
`finalize` - Interpolate labels on all images from the bag
`tensorflow` - Ro-configure the finalized dataset into the format expected by tensorflow model preprocessing stage.

--------------------------------------------------

### voyage_node (final ros node)

This is the final ros node that uses the trained model checkpoint and
applies it to every image seen in the raw camera topic. Required
reverse-engineering the inception model transformations to get the
input format to match the model checkpoint.

How I ran it:

```
TRAIN_DIR=/fast/models/voyage/1/
VOYAGE_DATA_DIR=/fast/datasets/voyage/tfr_single_ready/

cd ~/src/tensorflow-models/inception &&
bazel build inception/voyage_node &&
bazel-bin/inception/voyage_node --checkpoint_dir="${TRAIN_DIR}" \
```


--------------------------------------------------

### build_image_data

This is provided by the tf inception modeling pipeline. It takes the
output from the `tensorflow` step in bag_processing and converts all
the examples into sharded protobuf format.

How I ran it:

```
TRAIN_DIR=/fast/datasets/voyage/tfr_convertible/train
TEST_DIR=/fast/datasets/voyage/tfr_convertible/test
LABELS_FILE=/fast/datasets/voyage/tfr_convertible/labels.txt
OUTPUT_DIRECTORY=/fast/datasets/voyage/tfr_ready

cd ~/src/tensorflow-models/inception &&
bazel build inception/build_image_data &&
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${TEST_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=8 \
  --validation_shards=4 \
  --num_threads=4
```

--------------------------------------------------

### voyage_train

Use the pretrained inception v3 model to transfer-learn the new task
of stop-light detection. Requires downloading the inception v3
checkpoint data from google.

How I ran it:

```
mkdir -p $TRAIN_DIR
cd ~/src/tensorflow-models/inception &&
bazel build inception/voyage_train &&
bazel-bin/inception/voyage_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${VOYAGE_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1 \
  --num_gpus=2 \
  --max_steps=400
```

--------------------------------------------------

### inception/voyage_eval

Evaluates precision (accuracy) and top-5 recall on the held out test
set. This is also provided in the inception library with only slight
modifications for this dataset.

NOTE: I got about 94% accuracy at 80fps with this model.

```
TRAIN_DIR=/fast/models/voyage/1/
VOYAGE_DATA_DIR=/fast/datasets/voyage/tfr_ready/
EVAL_DIR=/fast/eval/voyage/1

mkdir -p $EVAL_DIR &&
bazel build inception/voyage_eval &&
bazel-bin/inception/voyage_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${VOYAGE_DATA_DIR}" \
  --subset=validation \
  --num_examples=8000 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once
```