#for DIM in 1024 2048
#do
#  python main.py --random_feature_dim $DIM
#done
#python main.py --random_feature_dim 32768 &
#python main.py --random_feature_dim 16384 &
python main.py --random_feature_dim 512 --env_id "Pendulum-v1" --reward_scale 0.2 --kernel_representation "random_feature"
python main.py --random_feature_dim 512 --env_id "Pendulum-v1" --reward_scale 0.2 --algo "SAC"


for SEED in 2 3 4
do
  python main.py --seed $SEED --random_feature_dim 1024
#  python main.py --seed 5
done