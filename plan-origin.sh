#!/bin/bash

conda init
source ~/.bashrc
conda activate tdmpc2

python tdmpc2/train_plan_origin.py
