"""Config file."""

from ml_collections import config_dict


def common_config():

  return dict(
      data_folder='synthetic/checkerboard',
      seed=1023,
      batch_size=128,
      total_train_steps=500000,
      learning_rate=1e-3,
      time_scale_factor=1000,
      time_duration=1.0,
      ema_decay=0.9999,
      lr_schedule='constant',
      diffuse_type='uniform',
      optimizer='adamw',
      transformer_norm_type='prenorm',
      uniform_rate_const=1.0,
      embed_dim=512,
      num_layers=3,
      log_every_steps=50,
      plot_every_steps=2000,
      save_every_steps=10000,
      plot_samples=1024,
      sampling_steps=1000,
      phase='train',
      model_init_folder='',
      save_root='',
      dtype='float32',
  )


def get_config():
  """Get config_dict."""
  cfg_dict = common_config()
  cfg_dict.update(dict(
      model_type='hollow',
      net_arch='bidir_transformer',
      readout='resnet',
      bidir_readout='res_concat',
      logit_type='direct',
      num_output_ffresiduals=2,
      loss_type='rm',
      eval_rounds=1,
      grad_norm=5.0,
      weight_decay=1e-6,
      num_heads=4,
      embed_dim=64,
      qkv_dim=64,
      mlp_dim=256,
      dropout_rate=0.0,
      learning_rate=1e-4,
      attention_dropout_rate=0.0,
      dropout_deterministic=False,
  ))
  config = config_dict.ConfigDict(initial_dictionary=cfg_dict)
  return config
