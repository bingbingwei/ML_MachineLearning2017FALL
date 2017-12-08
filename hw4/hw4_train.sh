python3 train_semi.py -train --model_out model_label.h5 --train_data $1
python3 train_semi.py -semi --model_data model_label.h5 --model_out model_semi.h5 --test_data $2
