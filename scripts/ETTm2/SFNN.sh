model_name=SFNN
datapath=ETT-small
dataset=ETTm2

for rid in {0..9} ; do
    for sl in 96 192 384 672 1344 ; do
        for pl in 96 192 336 720 ; do
            if [ $pl -eq 96 ]; then
                mixer="--mixer"
                wd=0.0005
                bs=64
            elif [ $pl -eq 192 ]; then
                mixer="--mixer"
                wd=0.001
                bs=64
            elif [ $pl -eq 336 ]; then
                mixer="--mixer"
                wd=0.0015
                bs=256
            else
                mixer=""
                wd=0.002
                bs=256
            fi
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 2 \
              $mixer \
              --need_norm --layernorm \
              --batch_size $bs \
              --train_epochs 100 \
              --weight_decay $wd \
              --dropout 0.7 \
              --loss_fn MAE \
              --learning_rate 0.0005
        done
    done
done
