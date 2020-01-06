#!/bin/bash

ps | grep python; pkill python;

mod='MOTOR'
blocks=17
python3 -W ignore ./utils.py --block_dura=${blocks} --task_modality=${mod} \
>./train_logs/${mod}_block${blocks}_gcn.log 2>&1

mod='ALL'
blocks=15
python3 -W ignore ./utils.py --block_dura=${blocks} --task_modality=${mod} \
>./train_logs/${mod}_block${blocks}_gcn.log 2>&1


