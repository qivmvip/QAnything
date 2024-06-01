#!/bin/bash

script_dir=$(dirname "$(readlink -f "$0")")
file="${script_dir}/.aliyun.dashscope.apikey.txt"
apikey="$(< "${file}" xargs echo -n)"

bash scripts/base_run.sh -s "M1mac" -w 4 -m 19530 -q 8777 -c -o -b 'https://dashscope.aliyuncs.com/compatible-mode/v1' -k "${apikey}" -n 'qwen-turbo' -l '4096'
