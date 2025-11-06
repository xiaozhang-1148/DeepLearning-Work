#!/bin/bash

version=$1

export LgEvalDir=$(pwd)/lgeval
export Convert2SymLGDir=$(pwd)/convert2symLG
export PATH=$PATH:$LgEvalDir/bin:$Convert2SymLGDir

for y in 'test/easy' 'test/medium' 'test/hard'
do
    echo '****************' start evaluating HME100K $y '****************'
    python scripts/test/test.py "$version" \
        --predict-split "$y" \
        --dataset-zip "HME100K_Dataset.zip" \
        --gpus 1
    echo
done