#for DIM in 1024 2048
#do
#  python main.py --random_feature_dim $DIM
#done
#python main.py --random_feature_dim 32768 &
#python main.py --random_feature_dim 16384 &
#python main.py --random_feature_dim 8192 &
#python main.py --random_feature_dim 4096 &
#python main.py --random_feature_dim 2048 &
python main.py --algo SAC
