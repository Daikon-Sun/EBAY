model_name=SFNN
datapath=exchange_rate
dataset=exchange_rate

for rid in {0..9} ; do
    for sl in 5 10 20 40 80 160 320 ; do
        for pl in 96 192 336 720 ; do
            if [ $pl -eq 96 ]; then
                mixer=""
                wd=0.0005
            elif [ $pl -eq 192 ]; then
                mixer=""
                wd=0.001
            elif [ $pl -eq 336 ]; then
                mixer=""
                wd=0.0015
            elif [ $pl -eq 720 ]; then
                mixer="--mixer"
                wd=0.002
            fi
            python -u run.py \
              --root_path ./dataset/"$datapath"/ \
              --data_path "$dataset".csv \
              --model_id "$dataset"_"$sl"_"$pl"_"$rid" \
              --model $model_name \
              --data $dataset \
              --seq_len $sl \
              --pred_len $pl \
              --n_layers 1 \
              $mixer \
              --batch_size 64 \
              --train_epochs 100 \
              --weight_decay $wd \
              --dropout 0.7 \
              --loss_fn MAE \
              --learning_rate 0.001 \
              --layernorm 0 \
              --need_norm 0 \
              --norm_len 1
        done
    done
done
