#!/bin/bash

ps | grep python; pkill python;

eigorder=$2
Korder=$3
eigstr='_eig'${eigorder}
if [ -z ${eigorder} ];then eigorder=0; eigstr='_fulleig'; fi
if [ -z ${Korder} ];then Korder=10;fi
##if [ ${eigorder} -eq 0 ];then eigstr='_fulleig'; fi

mod='MOTOR'
blocks=17
python3 -W ignore ./training.py --block_dura=${blocks} --task_modality=${mod} \
>./train_logs/${mod}_block${blocks}_gcn${eigstr}.log 2>&1

mod='WM'
blocks=35
python3 -W ignore ./training.py --block_dura=${blocks} --task_modality=${mod} \
>./train_logs/${mod}_block${blocks}_gcn${eigstr}.log 2>&1

mod='Language'
blocks=15
python3 -W ignore ./training.py --block_dura=${blocks} --task_modality=${mod} \
>./train_logs/${mod}_block${blocks}_gcn${eigstr}.log 2>&1

mod='ALL'
blocks=15
python3 -W ignore ./training.py --block_dura=${blocks} --task_modality=${mod} \
>./train_logs/${mod}_block${blocks}_gcn${eigstr}.log 2>&1


