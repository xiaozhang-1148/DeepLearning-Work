import argparse
import re
from typing import Set, List

from process_format_latex import _format_formula_for_display


def parse_caption_file_to_vocab(file_path: str) -> Set[str]:
    """
    读取caption.txt文件，提取所有latex标注中的唯一语法单元，返回一个集合。
    """
    unique_tokens = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            cleaned_line = line.split('\t')
            
            if not cleaned_line:
                continue
            
            parts = cleaned_line[1].replace('\n', '').split(' ')


            unique_tokens.update(parts)

    return unique_tokens


def save_vocab_to_txt(vocab_set: Set[str], file_path: str):
    """
    将语法单元集合以每行一个的形式保存到txt文件，并自动添加特殊标记。
    """
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    
    dataset_tokens = sorted(list(vocab_set))
    
    final_vocab = special_tokens + dataset_tokens
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for token in final_vocab:
            f.write(token + '\n')
    
    print(f"✅ 词汇表已成功保存！包含 {len(special_tokens)} 个特殊标记和 {len(dataset_tokens)} 个数据集标记，总计 {len(final_vocab)} 个。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从caption.txt文件中提取所有唯一的LaTeX语法单元，并生成一个包含特殊标记的完整词汇表文件。")

    parser.add_argument("-i", "--caption_path", type=str, required=True, help="包含所有标注的caption.txt文件路径")

    parser.add_argument("-o", "--output_path", type=str, required=True, help="保存最终词汇表的文件路径（例如：PosFormer-main/Pos_Former/datamodule/dic.txt）")

    args = parser.parse_args()

    try:
        vocab = parse_caption_file_to_vocab(args.caption_path)
        
        print(f"\n✅ 解析完成！从数据集中共找到 {len(vocab)} 个唯一的语法单元。")
        
        print(f"\n正在将完整词汇表保存到: {args.output_path} ...")
        save_vocab_to_txt(vocab, args.output_path)
        
    except FileNotFoundError:
        print(f"\n❌ 错误：文件不存在 -> {args.caption_path}")
    except Exception as e:
        print(f"\n❌ 发生未知错误：{str(e)}")