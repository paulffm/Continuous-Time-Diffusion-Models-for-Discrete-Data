#!/bin/bash

data_name="checkerboard"
data_root="${HOME}/data/sddm/synthetic/${data_name}"

python -m sddm.synthetic.data.main_datadump \
  --data_root="${data_root}" \
  --data_config="data_config.py" \
  --num_samples=10000000 \
  --data_config.data_name="${data_name}" \
