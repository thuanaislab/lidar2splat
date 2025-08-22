#! /bin/bash  

python -m pip install virtualenv
python -m virtualenv trans-env
source trans-env/bin/activate
pip install -r requirements.txt

deactivate