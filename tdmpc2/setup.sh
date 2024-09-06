# !/bin/bash

cd ../../GenerativeRL_Preview
pip install -e .

cd ../Metaworld
pip install -e .

pip install pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install hydra-core==1.3.2 hydra-submitit-launcher==1.2.0
pip install dm-control==1.0.14
pip install tensordict==0.5.0
pip install -r /mnt/nfs/chenxinyan/tdmpc2/requirements.txt
