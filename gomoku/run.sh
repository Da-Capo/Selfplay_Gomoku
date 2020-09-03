#!/bin/bash
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple
die () {
    echo >&2 "$@"
    exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

ENVIRONMENT=$1
AGENT=$2
NUM_ACTORS=$3
NODE_ID_ST=$4
NODE_NUM_ACTORS=$5
NODE_TYPE=$6
SERVER_ADDRESS=$7

shift 3

export PYTHONPATH=$PYTHONPATH:/
export TF_FORCE_GPU_ALLOW_GROWTH=true

ACTOR_BINARY="CUDA_VISIBLE_DEVICES='' python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor";
LEARNER_BINARY="python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner";
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

tmux new-session -d -t seed_rl
mkdir -p /tmp/seed_rl
cat >/tmp/seed_rl/instructions <<EOF
Welcome to the SEED local training of ${ENVIRONMENT} with ${AGENT}.
SEED uses tmux for easy navigation between different tasks involved
in the training process. To switch to a specific task, press CTRL+b, [tab id].
You can stop training at any time by executing '../stop_local.sh'
EOF

tmux send-keys clear
tmux send-keys KPEnter
tmux send-keys "cat /tmp/seed_rl/instructions"
tmux send-keys KPEnter
tmux send-keys "python3 check_gpu.py 2> /dev/null"
tmux send-keys KPEnter
tmux send-keys "../stop_local.sh"
PARAM='--batch_size 256 --use_lnet 0 --inference_batch_size 16 --unroll_length 20 --learning_rate 0.000048'
PARAM=$PARAM' --discounting 0.995 --lambda_ 0.95 --baseline_cost 0.5 --entropy_cost 0.0001'

if [ "$NODE_TYPE" == "server" ];then

    tmux new-window -d -n learner
    COMMAND='rm /tmp/agent/* -Rf; '"${LEARNER_BINARY}"' --logtostderr --pdb_post_mortem '"$PARAM"' --server_address='"${SERVER_ADDRESS}"' --num_actors='"${NUM_ACTORS}"''
    echo $COMMAND
    tmux send-keys -t "learner" "$COMMAND" ENTER

    tmux new-window -d -n "actor_eval"
    COMMAND="CUDA_VISIBLE_DEVICES='' python3 ../${ENVIRONMENT}/actor_eval.py"
    tmux send-keys -t "actor_eval" "$COMMAND" ENTER

    # tmux new-window -d -n "actor_eval_embryo"
    # COMMAND="CUDA_VISIBLE_DEVICES='' python3 ../${ENVIRONMENT}/actor_eval_embryo.py"
    # tmux send-keys -t "actor_eval_embryo" "$COMMAND" ENTER

    tmux new-window -d -n "tensorboard"
    COMMAND="tensorboard --logdir=/tmp/agent --bind_all"
    tmux send-keys -t "tensorboard" "$COMMAND" ENTER
fi

for ((id=$NODE_ID_ST; id<$(($NODE_ID_ST+$NODE_NUM_ACTORS)); id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND=''"${ACTOR_BINARY}"' --logtostderr --pdb_post_mortem '"$PARAM"' --server_address='"${SERVER_ADDRESS}"' --num_actors='"${NUM_ACTORS}"' --task='"${id}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done

tmux set -g mouse on
tmux attach -t seed_rl
