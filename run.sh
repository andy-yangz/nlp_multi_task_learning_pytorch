echo "Running Model"

for nhid in 64 128 256
do
    python main.py --data './data' \
                --emsize 256 \
                --nlayers 1 \
                --nhid $nhid \
                --batch_size 128 \
                --seq_len 15 \
                --cuda \
                --epochs 300 \
                --log_interval 10 \
                --save './result/pos_model'
done

for nlayers in 2 3 
do
    python main.py --data './data' \
                --emsize 256 \
                --nlayers $nlayers \
                --nhid 128 \
                --batch_size 128 \
                --seq_len 15 \
                --cuda \
                --epochs 300 \
                --log_interval 10 \
                --save './result/pos_model'
done

for dropout in 0.1 0.4 0.6
do 
    python main.py --data './data' \
        --emsize 256 \
        --nlayers 1 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 15 \
        --dropout $dropout \
        --cuda \
        --epochs 300 \
        --log_interval 10 \
        --save './result/pos_model'
done

echo "Bidirection"
python main.py --data './data' \
        --emsize 256 \
        --nlayers 1 \
        --nhid 128 \
        --batch_size 128 \
        --seq_len 15 \
        --dropout 0.4
        --cuda \
        --bi \
        --epochs 150 \
        --log_interval 10 \
        --save './result/pos_model'

# for emsize in 128 512 
# do
    # python main.py --data './data' \
    #             --emsize $emsize \
    #             --nlayers 1 \
    #             --nhid 128 \
    #             --batch_size 128 \
    #             --seq_len 15 \
    #             --cuda \
    #             --epochs 150 \
    #             --log_interval 10 \
    #             --save './result/pos_model'
# done

# for nlayers in 1 2 3 
# do
#     python main.py --data './data' \
#                 --emsize 256 \
#                 --nlayers $nlayers \
#                 --nhid 128 \
#                 --batch_size 128 \
#                 --seq_len 15 \
#                 --cuda \
#                 --epochs 150 \
#                 --log_interval 10 \
#                 --save './result/pos_model'
# done

# for nhid in 256 512
# do
#     python main.py --data './data' \
#                 --emsize 256 \
#                 --nlayers 1 \
#                 --nhid $nhid \
#                 --batch_size 128 \
#                 --seq_len 15 \
#                 --cuda \
#                 --epochs 150 \
#                 --log_interval 10 \
#                 --save './result/pos_model'
# done

# for seq_len in 10 20
# do
#     python main.py --data './data' \
#             --emsize 256 \
#             --nlayers 1 \
#             --nhid 128 \
#             --batch_size 128 \
#             --seq_len $seq_len \
#             --cuda \
#             --epochs 150 \
#             --log_interval 10 \
#             --save './result/pos_model'
# done

# echo "Bidirection"
# python main.py --data './data' \
#         --emsize 256 \
#         --nlayers 1 \
#         --nhid 128 \
#         --batch_size 128 \
#         --seq_len 15 \
#         --cuda \
#         --bi \
#         --epochs 150 \
#         --log_interval 10 \
#         --save './result/pos_model'

