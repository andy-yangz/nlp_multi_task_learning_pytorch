echo "Running Model on samples"

python main.py --data './data/sample' \
            --emsize 20 \
            --npos_layers 1 \
            --nchunk_layers 2 \
            --train_mode 'Joint' \
            --nhid 20 \
            --batch_size 2 \
            --epochs 10 \
            --seq_len 10 \
            --cuda \
            --log_interval 2
