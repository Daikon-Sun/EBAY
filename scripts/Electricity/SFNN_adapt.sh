model_name=SFNN
datapath=electricity
dataset=electricity

sl=168
pl=1
rid=$1
mode=adapt

br=0
er=0.25
bs=1024

for ep in 10 20 40 80 ; do
    for lr in 1e-4 3e-4 1e-3 3e-3 ; do
        for adapt_iters in 0 1 2 4 8 16 ; do
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --mode $mode \
              --adapt_iters $adapt_iters \
              --test_batch_size 168 \
              --adapt_lr $lr \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 3 \
              --batch_size 32 \
              --train_epochs $ep \
              --weight_decay 0.00001 \
              --dropout 0.5 \
              --loss_fn MAE \
              --learning_rate 0.0005 \
              --beg_ratio $br \
              --end_ratio $er
        done
    done
done
