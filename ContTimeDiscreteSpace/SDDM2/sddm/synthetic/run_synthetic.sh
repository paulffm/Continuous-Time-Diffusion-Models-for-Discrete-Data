#!/bin/bash

data_name="checkerboard"
data_root="${HOME}/data/sddm"
save_root="${HOME}/results/sddm/${config_name?}/${data_name}"

python -m sddm.synthetic.main_binary_graycode \
  --data_root="${data_root}" \
  --config="config/${config_name}.py" \
  --config.data_folder="synthetic/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \
