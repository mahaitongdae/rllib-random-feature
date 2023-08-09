#for DIM in 1024 2048
#do
#  python main.py --random_feature_dim $DIM
#done
#python main.py --random_feature_dim 32768 &
#python main.py --random_feature_dim 16384 &
python main.py --seed 2
python main.py --seed 3
python main.py --seed 4
python main.py --seed 5
