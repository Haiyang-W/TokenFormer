<!--
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->



# Code Structure
This code is based on the awesome codebase of Maxtext for training LMs. Please see the original README [here](https://github.com/AI-Hypercomputer/maxtext/tree/main).

- The tokenformer is implemented in `Maxtext/layers/tokenformer.py`
- We use [Grain dataloader](https://github.com/google/grain) for its determinism when training large models. See `MaxText/input_pipeline/_grain_data_processing.py`
- The config for running all TokenFormer experiments is `MaxText/configs/tokenformer.yml`
- We add a `create_skipped_iterator()` in train.py to automatically resume when training collapse due to the dataloading
- Parameter reusing code for model scaling is implemented in `MaxText/max_utils.py`

# Prerequisite:
We use [Google Cloud Platform](https://cloud.google.com/?hl=en) for the main results in the paper.
- Follow the official MaxText README to install the env.
- Dataset: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext). Download the original dataset and then convert it to .arrayrecord (as we are using Grain dataloader). We reserve 5% of the training data for computing the val performance.
- We use Cloud Storage buckets for storing the checkpoints and the data.

# Training
## TokenFormer with model scaling
```
python3 MaxText/train.py MaxText/configs/tokenformer.yml run_name=tokenformer124_steps600k base_output_directory=gs://tokenformer_gcp steps=600000 dataset_type=grain grain_train_files=/tmp/gcsfuse/openwebtext_arrayrecord/train* tokenizer_path=EleutherAI/gpt-neox-20b grain_worker_count=8 eval_interval=10000 grain_eval_files=/tmp/gcsfuse/openwebtext_arrayrecord/validation* enable_checkpointing=True per_device_batch_size=32 eval_per_device_batch_size=32 checkpoint_period=10000 logits_via_embedding=True normalize_embedding_logits=True slot_num=576 drop_path=0 init_weights_seed=42
```


```
python3 MaxText/train.py MaxText/configs/tokenformer.yml run_name=tokenformer354_steps120k base_output_directory=gs://tokenformer_gcp steps=120000 dataset_type=grain grain_train_files=/tmp/gcsfuse/openwebtext_arrayrecord/train* tokenizer_path=EleutherAI/gpt-neox-20b grain_worker_count=8 eval_interval=10000 grain_eval_files=/tmp/gcsfuse/openwebtext_arrayrecord/validation* enable_checkpointing=True per_device_batch_size=32 eval_per_device_batch_size=32 checkpoint_period=2000 logits_via_embedding=True normalize_embedding_logits=True slot_num=2140 load_parameters_path=gs://tokenformer_gcp/tokenformer124_steps600k/checkpoints/590000/items load_parameters_path_slot_number=576 drop_path=0 init_weights_seed=42
```

## Transformer baseline
```
python3 MaxText/train.py MaxText/configs/gpt2.yml run_name=transformer124_steps600k base_output_directory=gs://tokenformer_gcp steps=600000 dataset_type=grain grain_train_files=/tmp/gcsfuse/openwebtext_arrayrecord/train* tokenizer_path=EleutherAI/gpt-neox-20b grain_worker_count=8 eval_interval=10000 grain_eval_files=/tmp/gcsfuse/openwebtext_arrayrecord/validation* enable_checkpointing=True per_device_batch_size=32 eval_per_device_batch_size=32 checkpoint_period=2000 logits_via_embedding=True normalize_embedding_logits=True init_weights_seed=42
```