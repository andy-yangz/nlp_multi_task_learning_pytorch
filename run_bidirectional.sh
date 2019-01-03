echo "Running Model"

for emsize in 256 300
do
    echo "Embedding size $emsize"
    echo "POS"
    python main.py --data $1 \
            --emsize $emsize \
            --npos_layers 2 \
            --nchunk_layers 0 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'POS' \
            --epochs 300 \
            --log_interval 20 \
            --bi \
            --save "./result/pos_model_embed_$emsize"

    echo "Chunk"
    python main.py --data $1 \
            --emsize $emsize \
            --npos_layers 0 \
            --nchunk_layers 2 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'Chunk' \
            --epochs 300 \
            --log_interval 20 \
            --bi \
            --save "./result/chunk_model_embed_$emsize"

    echo "Joint Training on the same level"
    python main.py --data $1 \
            --emsize $emsize \
            --npos_layers 2 \
            --nchunk_layers 2 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'Joint' \
            --epochs 300 \
            --log_interval 20 \
            --bi \
            --save "./result/joint_same_embed_$emsize"

    echo "Joint Training on the different level"
    python main.py --data $1 \
            --emsize $emsize \
            --npos_layers 1 \
            --nchunk_layers 2 \
            --nhid 128 \
            --batch_size 128 \
            --seq_len 10 \
            --cuda \
            --train_mode 'Joint' \
            --epochs 300 \
            --log_interval 20 \
            --bi \
            --save "./result/joint_diff_embed_$emsize"
done

echo "Using Pre-trained embeddings"

echo "Embedding size $emsize"
echo "POS"
python main.py --data $1 \
        --npos_layers 2 \
        --nchunk_layers 0 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 10 \
        --cuda \
        --train_mode 'POS' \
        --epochs 300 \
        --log_interval 20 \
        --pretrained_embeddings \
        --bi \
        --save "./result/pos_model_glove_embed"

echo "Chunk"
python main.py --data $1 \
        --npos_layers 0 \
        --nchunk_layers 2 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 10 \
        --cuda \
        --train_mode 'Chunk' \
        --epochs 300 \
        --log_interval 20 \
        --pretrained_embeddings \
        --bi \
        --save "./result/chunk_model_glove_embed"

echo "Joint Training on the same level"
python main.py --data $1 \
        --npos_layers 2 \
        --nchunk_layers 2 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 10 \
        --cuda \
        --train_mode 'Joint' \
        --epochs 300 \
        --log_interval 20 \
        --pretrained_embeddings \
        --bi \
        --save "./result/joint_same_glove_embed"

echo "Joint Training on the different level"
python main.py --data $1 \
        --npos_layers 1 \
        --nchunk_layers 2 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 10 \
        --cuda \
        --train_mode 'Joint' \
        --epochs 300 \
        --log_interval 20 \
        --pretrained_embeddings \
        --bi \
        --save "./result/joint_diff_glove_embed"