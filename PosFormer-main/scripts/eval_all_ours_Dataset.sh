#!/bin/bash

version=$1

export LgEvalDir=$(pwd)/lgeval
export Convert2SymLGDir=$(pwd)/convert2symLG
export PATH=$PATH:$LgEvalDir/bin:$Convert2SymLGDir

for y in 'test/easy' 'test/medium' 'test/hard'
do
    echo '****************' start evaluating ours_Dataset $y '****************'
    python scripts/evaluate.py --path "$version" \
        --data-split "$y" \
        --dataset-zip "data/raw/ours_Dataset.zip" \
        --gpus 1
    echo
done