wget https://www.dropbox.com/s/czi5gkluofwtqoo/model.h5?dl=1
mv model.h5?dl=1 model.h5
python3 test.py --model_data model.h5 --sub_data $1
