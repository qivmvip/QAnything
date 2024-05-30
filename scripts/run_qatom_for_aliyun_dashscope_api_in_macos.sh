#!/bin/bash

#!/bin/bash

# 声明一个数组
declare -a lines

# 使用文件描述符打开文件以读取模式
while IFS= read -r line
do
  # 将行追加到数组中
  lines+=("$line")
done < "/path/to/your/file.txt"

# 现在，整个文件的内容按行存储在数组 ${lines[@]} 中
# 您可以使用 ${lines[0]}、${lines[1]} 等方式访问每个元素。



bash scripts/base_run.sh -s "M1mac" -w 4 -m 19530 -q 8777 -c -o -b 'https://dashscope.aliyuncs.com/compatible-mode/v1' -k 'sk-xxx' -n 'qwen-turbo' -l '4096'
