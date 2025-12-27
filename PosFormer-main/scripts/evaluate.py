import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))  # Add src to path

import os
import typer
import zipfile
from posformer.datamodule import CROHMEDatamodule
from posformer.core.processors import LitPosFormer
from pytorch_lightning import Trainer, seed_everything
from typing import Optional

seed_everything(42)

def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range (m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]

def main(
    path: str = typer.Option(..., "--path", "-p", help="包含 .ckpt 文件的模型检查点文件夹路径。"),
    data_split: str = typer.Option("test", "--data-split", help="要预测的数据划分。"),
    gpus: int = typer.Option(1, "--gpus", help="使用的 GPU 数量。"),
    dataset_zip: str = typer.Option("--dataset-zip", help="数据集 zip 文件的路径。")
):
    """
    对指定的模型版本和数据集划分进行测试和评估。
    """
    # 动态构建检查点路径
    ckp_folder = path
    
    # 查找最新的 .ckpt 文件
    try:
        fnames = [f for f in os.listdir(ckp_folder) if f.endswith('.ckpt')]
        fnames.sort(key=lambda x: os.path.getmtime(os.path.join(ckp_folder, x)), reverse=True)
        assert len(fnames) > 0, "No checkpoint file found."
        ckp_path = os.path.join(ckp_folder, fnames[0])
        print(f"--- Found checkpoint: {fnames[0]} ---")
    except (FileNotFoundError, AssertionError) as e:
        print(f"Error: Could not find checkpoint in {ckp_folder}. {e}")
        sys.exit(1)

    trainer = Trainer(logger=False, gpus=gpus)

    dm = CROHMEDatamodule(
        zipfile_path=dataset_zip,
        eval_batch_size=1,
        num_workers=16
    )

    model = LitPosFormer.load_from_checkpoint(ckp_path)
    print(f"--- Running test on '{data_split}' dataset... ---")
    trainer.test(model, datamodule=dm)
    
    print(f"--- Calculating Expression Rate... ---")
    caption = {}
    with zipfile.ZipFile(dataset_zip) as archive:
        caption_path = f"data/{data_split}/caption.txt"
        try:
            with archive.open(caption_path, "r") as f:
                caption_lines = [line.decode('utf-8').strip() for line in f.readlines()]
                for caption_line in caption_lines:
                    if not caption_line:
                        continue
                    caption_parts = caption_line.split('\t')
                    caption_file_name = caption_parts[0]
                    caption_string = caption_parts[1]
                    caption[caption_file_name] = caption_string
        except KeyError:
            print(f"Error: Ground truth file not found in zip: {caption_path}")
            sys.exit(1)

    with zipfile.ZipFile("result.zip") as archive:
        exprate=[0,0,0,0]
        file_list = archive.namelist()
        txt_files = [file for file in file_list if file.endswith('.txt')]
        for txt_file in txt_files:
            file_name = os.path.basename(txt_file).replace('.txt', '')
            with archive.open(txt_file) as f:
                lines = f.readlines()
                if len(lines) < 2:
                    continue
                pred_string = lines[1].decode('utf-8').strip().replace('$', '')
                if file_name in caption:
                    caption_string = caption[file_name]
                else:
                    print(f"Warning: {file_name} not found in caption file, skipping.")
                    continue
                
                caption_parts = caption_string.strip().split()
                pred_parts = pred_string.strip().split()

                if caption_string == pred_string:
                    exprate[0]+=1
                
                error_num=cal_distance(pred_parts,caption_parts)
                if error_num > 0 and error_num <= 3:
                    exprate[error_num]+=1

        tot = len(txt_files)
        if tot == 0:
            print("No results found in result.zip to evaluate.")
            return
             
        exprate_final=[]
        for i in range(1,5):
            exprate_final.append(100*sum(exprate[:i])/tot)
        print(f"--- Results for '{data_split}' dataset ---")
        print(f"Total samples: {tot}")
        print(f"Correct (ExpRate): {exprate_final[0]:.2f}%")
        print(f"Correct <= 1 error: {exprate_final[1]:.2f}%")
        print(f"Correct <= 2 errors: {exprate_final[2]:.2f}%")
        print(f"Correct <= 3 errors: {exprate_final[3]:.2f}%")

if __name__ == "__main__":
    typer.run(main)
