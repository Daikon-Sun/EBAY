model_name=SFNN
datapath=solar
dataset=solar

sl=144
pl=144
rid=$1

for br in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 ; do
    wd=0.001
    wd=0
    lf="MSE"
    dr=0.1
    dr=0
    lr=3e-3
    bs=1024
    er=0.6
    echo $br $er
    python -u run.py \
      --root_path ./dataset/"$datapath"/ \
      --data_path "$dataset".csv \
      --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
      --model $model_name \
      --data $dataset \
      --seq_len $sl \
      --pred_len $pl \
      --n_layers 2 \
      --mixer \
      --batch_size $bs \
      --train_epochs 200 \
      --weight_decay $wd \
      --dropout $dr \
      --loss_fn $lf \
      --learning_rate $lr\
      --min_lr 5e-5 \
      --beg_ratio $br \
      --end_ratio $er
done
