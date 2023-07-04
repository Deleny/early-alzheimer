---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': Mild_Demented
          '1': Moderate_Demented
          '2': Non_Demented
          '3': Very_Mild_Demented
  splits:
  - name: train
    num_bytes: 22560791.2
    num_examples: 5120
  - name: test
    num_bytes: 5637447.08
    num_examples: 1280
  download_size: 28289848
  dataset_size: 28198238.28
---
# Dataset Card for "Alzheimer_MRI"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)