import re
import argparse
import sys

all = [
    "_format_formula_for_display",
    "process_caption_file",
]

def _is_cjk_char(char: str) -> bool:
    """
    检查字符是否为中日韩（CJK）统一表意文字。
    """
    return '\u4e00' <= char <= '\u9fff'

def _format_formula_for_display(formula: str) -> str:
    """Format a tokenized formula for better display.
    
    - Join Chinese characters without spaces
    - Format LaTeX commands more readably
    """
    # Check if formula contains Chinese characters
    has_chinese = any(_is_cjk_char(c) for c in formula)
    
    if has_chinese:
        # Process formula with Chinese characters
        tokens = formula.split()
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Handle LaTeX commands
            if token.startswith('\\'):
                result.append(token)
            # Add \text{} for individual Chinese characters as well as consecutive ones
            elif any(_is_cjk_char(c) for c in token):
                if i + 1 < len(tokens) and any(_is_cjk_char(c) for c in tokens[i+1]):
                    # Handle consecutive Chinese characters by joining them
                    chinese_phrase = [token]
                    j = i + 1
                    while j < len(tokens) and any(_is_cjk_char(c) for c in tokens[j]):
                        chinese_phrase.append(tokens[j])
                        j += 1
                    chinese_str = ''.join(chinese_phrase)
                    result.append(f"\\text{{{chinese_str}}}")
                    i = j - 1
                else:
                    # Handle single Chinese character or token containing Chinese
                    result.append(f"\\text{{{token}}}")
            else:
                result.append(token)
            i += 1
        
        formatted = ' '.join(result)
    else:
        # For non-Chinese formulas, just make LaTeX commands more readable
        formatted = formula
    
    # Improve readability of common LaTeX structures
    
    # Add thin spaces after commas
    formatted = formatted.replace(',', ', ')
    formatted = formatted.replace('\\delete', '\\text{\\{deleted\\}}')
    
    # Common LaTeX environments - format them properly
    environments = [
        "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", 
        "cases", "array", "aligned", "align", "align*", "eqnarray", "eqnarray*",
        "equation", "equation*", "gather", "gather*", "multline", "multline*",
        "split", "tabular"
    ]
    
    for env in environments:
        # Fix spacing in begin and end environment commands
        formatted = formatted.replace(f'\\begin {{ {env} }}', f'\\begin{{{env}}}')
        if env == "array":
            formatted = re.sub(r'{array}\s*{\s*([^{}]+)\s*}', lambda m: '{array}{' + m.group(1).replace(' ', '') + '}', formatted)
        formatted = formatted.replace(f'\\end {{ {env} }}', f'\\end{{{env}}}')
    
    # Improve spacing around brackets in general
    bracket_pairs = [
        ('\\left', '\\right'),
        ('\\bigg', '\\bigg'),
        ('\\Big', '\\Big'),
        ('\\big', '\\big')
    ]
    
    for left, right in bracket_pairs:
        for bracket in ['(', '[', '\\{', '|', '\\langle', '\\lceil', '\\lfloor']:
            # Add proper spacing after left bracket commands
            formatted = formatted.replace(f'{left} {bracket}', f'{left}{bracket}')
        
        for bracket in [')', ']', '\\}', '|', '\\rangle', '\\rceil', '\\rfloor']:
            # Add proper spacing before right bracket commands
            formatted = formatted.replace(f'{right} {bracket}', f'{right}{bracket}')
    
    # Fix common fraction notations
    formatted = re.sub(r'\\frac\s*{\s*([^{}]+)\s*}\s+{\s*([^{}]+)\s*}', r'\\frac{\1}{\2}', formatted)
    
    # Fix \text commands
    formatted = re.sub(r'\\text\s+{\s*([^{}]+)\s*}', r'\\text{\1}', formatted)
    formatted = re.sub(r'\\textcircled\s+{\s*([^{}]+)\s*}', r'\\textcircled{\1}', formatted)
    
    # Fix spacing in subscripts and superscripts (also remove any preceding spaces)
    formatted = re.sub(r'\s*_\s*{\s*([^{}]+)\s*}', r'_{\1}', formatted)
    formatted = re.sub(r'\s*\^\s*{\s*([^{}]+)\s*}', r'^{\1}', formatted)
    
    # Additional cleanup of curly braces - remove spaces inside simple braces
    formatted = re.sub(r'{\s+(\S+)\s+}', r'{\1}', formatted)
    
    # Special handling for array/matrix entries with ampersands
    formatted = re.sub(r'\s*&\s*', r' & ', formatted)
    
    # More aggressive cleaning of spaces inside braces
    # First, handle nested braces carefully by working on innermost braces first
    prev_formatted = ""
    while prev_formatted != formatted:
        prev_formatted = formatted
        # Remove trailing spaces inside braces
        formatted = re.sub(r'{([^{}]*?)\s+}', r'{\1}', formatted)
        # Remove leading spaces inside braces
        formatted = re.sub(r'{\s+([^{}]*?)}', r'{\1}', formatted)
    
    # Clean up spaces around operators
    for op in ['+', '-', '=', '<', '>']:
        formatted = re.sub(f'\\s*{re.escape(op)}\\s*', f' {op} ', formatted)
    
    # Handle LaTeX comparison operators separately
    formatted = re.sub(r'\s*\\leq\s*', r' \\leq ', formatted)
    formatted = re.sub(r'\s*\\geq\s*', r' \\geq ', formatted)
    formatted = re.sub(r'\s*\\neq\s*', r' \\neq ', formatted)
    
    # NEW: tighten spacing around LaTeX commands and braces (e.g. "\\frac {a}{b}" -> "\\frac{a}{b}")
    # 1. Remove spaces between a command (\\command) and the following opening brace
    formatted = re.sub(r'(\\[A-Za-z]+)\s+{', r'\1{', formatted)
    # 2. Remove spaces between a closing brace and the next opening brace
    formatted = re.sub(r'}\s+{', r'}{', formatted)
    # 3. Remove leading/trailing spaces immediately inside braces – this is safe even with nested braces
    formatted = re.sub(r'{\s+', '{', formatted)
    formatted = re.sub(r'\s+}', '}', formatted)
    
    # Clean up excessive spaces
    formatted = re.sub(r'\s+', ' ', formatted)
    formatted = formatted.strip()
    
    return formatted

def process_caption_file(input_path: str, output_path: str):
    """
    读取一个caption文件，对每一行的LaTeX进行格式化，并写入到新文件。
    """
    print(f"--- 开始处理文件 ---")
    print(f"读取输入文件: {input_path}")
    print(f"写入输出文件: {output_path}")
    
    lines_processed = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # 去除行尾的换行符
                stripped_line = line.strip()
                if not stripped_line:
                    continue  # 跳过空行

                # 按第一个制表符分割，确保公式中的制表符不会被错误分割
                parts = stripped_line.split('\t', 1)
                
                if len(parts) == 2:
                    image_name, formula = parts
                    # 调用核心函数格式化LaTeX公式
                    formatted_formula = _format_formula_for_display(formula)
                    # 将格式化后的结果以 "图片名\t公式" 的格式写回文件
                    outfile.write(f"{image_name}\t{formatted_formula}\n")
                    lines_processed += 1
                else:
                    # 如果某一行不符合 "图片名\t公式" 的格式，直接原样写入
                    outfile.write(stripped_line + '\n')
                    print(f"警告: 第 {lines_processed + 1} 行格式不正确，已原样写入: '{stripped_line}'")

        print(f"\n--- 处理完成 ---")
        print(f"总共处理并格式化了 {lines_processed} 行。")

    except FileNotFoundError:
        print(f"\n错误: 输入文件未找到 '{input_path}'", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"\n错误: 读写文件时发生错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="读取一个caption文件，格式化其中的LaTeX公式，并保存到新文件。"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="输入的caption文件路径 (例如: data/caption.txt)"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="格式化后输出的caption文件路径 (例如: data/caption_formatted.txt)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主处理函数
    process_caption_file(args.input, args.output)