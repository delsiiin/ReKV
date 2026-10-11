conda create -n rekv python=3.12 -y
conda activate rekv

pip install -r requirements.txt
pip install -e . -c requirements.txt

cd model/longva
pip install -e . -c ../../requirements.txt
