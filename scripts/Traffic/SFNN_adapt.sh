model_name=SFNN
datapath=traffic
dataset=traffic

sl=1344
pl=1
rid=$1
mode=adapt

rid=$1
br=0
er=0.25

for ep in 10 20 40 80 ; do
    for lr in 1e-4 3e-4 1e-3 3e-3 ; do
        for adapt_iters in 0 1 2 4 8 16 ; do
            echo $br $er
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --mode $mode \
              --adapt_iters $adapt_iters \
              --adapt_lr $lr \
              --test_batch_size 168 \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 3 \
              --batch_size 32 \
              --train_epochs $ep \
              --weight_decay 0.00005 \
              --dropout 0.1 \
              --loss_fn MSE \
              --learning_rate 0.0004 \
              --min_lr 1e-5 \
              --beg_ratio $br \
              --end_ratio $er
        done
    done
done
