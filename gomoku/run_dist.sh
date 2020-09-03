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


set -e
die () {
    echo >&2 "$@"
    exit 1
}
No_NODE=$1

export ENVIRONMENT=gomoku
export AGENT=vtrace
export CONFIG=$ENVIRONMENT
export NUM_ACTORS=200

if [ $No_NODE == 0 ];then
  NODE_ID_ST=0
  NODE_NUM_ACTORS=40
  NODE_TYPE='server'
  SERVER_ADDRESS='localhost:8686'
  echo $ENVIRONMENT $AGENT $NUM_ACTORS $NODE_ID_ST $NODE_NUM_ACTORS $NODE_TYPE $SERVER_ADDRESS
  docker run --gpus all --entrypoint /seed_rl/$ENVIRONMENT/run.sh -ti -it \
        -p 6006:6006 \
        -v $(pwd):/seed_rl \
        -v $(pwd)/$ENVIRONMENT/agent/`date +%s`:/tmp/agent \
        -e HOST_PERMS="$(id -u):$(id -g)" --name seed --rm \
        seed_rl:gym $ENVIRONMENT $AGENT $NUM_ACTORS $NODE_ID_ST $NODE_NUM_ACTORS $NODE_TYPE $SERVER_ADDRESS
fi

if [ $No_NODE == 1 ];then
  NODE_ID_ST=0
  NODE_NUM_ACTORS=0
  NODE_TYPE='client'
  SERVER_ADDRESS='localhost:8686'
  docker run --gpus all --entrypoint "" -it  \
        -v $(pwd):/seed_rl \
        -v $(pwd)/$ENVIRONMENT/agent/`date +%s`:/tmp/agent \
        -e HOST_PERMS="$(id -u):$(id -g)" --name seed --rm \
        seed_rl:gym bash
fi