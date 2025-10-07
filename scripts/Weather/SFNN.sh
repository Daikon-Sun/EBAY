model_name=SFNN
datapath=weather
dataset=weather

for rid in {0..9} ; do
    for sl in 144 288 576 1008 ; do
        for pl in 96 192 336 720 ; do
            if [ $pl -eq 96 ] ; then
                norms="--need_norm --layernorm"
            elif [ $pl -eq 192 ] ; then
                norms="--layernorm"
            elif [ $pl -eq 336 ] ; then
                norms=""
            elif [ $pl -eq 720 ] ; then
                norms=""
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
              --mixer \
              $norms \
              --batch_size 128 \
              --train_epochs 100 \
              --weight_decay 0.0005 \
              --dropout 0.4 \
              --loss_fn MAE \
              --learning_rate 0.001
        done
    done
done
