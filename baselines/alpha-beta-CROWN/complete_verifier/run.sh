#!/bin/bash
cd /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier

#python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_5x50.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_5x50_nvidia3090_check_log_BAB_refine_mip_timeout2000_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt
#
#python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_5x80.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_5x80_nvidia3090_check_log_BAB_refine_mip_timeout2000_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt
#
#python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_6x100.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_6x100_nvidia3090_check_log_BAB_refine_mip_timeout2000_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt
#
#python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_9x100.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_9x100_nvidia3090_check_log_BAB_refine_mip_timeout2000_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt
#
#python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_9x200.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_9x200_nvidia3090_check_log_BAB_refine_mip_timeout2000_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt

python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_10x80_01.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_10x80_nvidia3090_check_log_BAB_refine_mip_timeout15_percentage01_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt

python abcrown.py --config /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/yaml_settings/pytorch_model_with_batch_vnnlib_mnist_10x80_05.yaml | tee /home/$(whoami)/mymac_remotedir/alpha-beta-CROWN/complete_verifier/my_experiment/my_logs/mnist_10x80_nvidia3090_check_log_BAB_refine_mip_timeout15_percentage05_default_setting_$(date '+%Y-%m-%d_%H-%M-%S').txt


