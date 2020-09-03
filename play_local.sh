
CHCKPOINT_PATH=$1
CHCKPOINT_DIR=${CHCKPOINT_PATH%/*}
echo ${CHCKPOINT_DIR} ${CHCKPOINT_PATH}

docker run --gpus all --entrypoint "" -it  \
        -v $(pwd):/seed_rl \
        -v ${CHCKPOINT_DIR}:/${CHCKPOINT_DIR} \
        -e HOST_PERMS="$(id -u):$(id -g)" \
        -e PYTHONPATH=$PYTHONPATH:/ \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        -e CUDA_VISIBLE_DEVICES='' \
        -w /seed_rl/gomoku \
        --name seed_play --rm \
        seed_rl:gym python actor_ckpt.py --init_checkpoint ${CHCKPOINT_PATH} --num_actors 1