import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from PIL import Image, ImageFont, ImageDraw
Image.MAX_IMAGE_PIXELS = 200000000
from matplotlib import rcParams
from itertools import repeat
import matplotlib as mpl
import io
import os
import re
import sys
import cv2
import concurrent.futures
import functools
from functools import lru_cache
import argparse
import tqdm
import random

__all__ = ["render_latex"]


mpl.use('pgf')

""" LaTeX渲染包名 """
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.rcfonts': False,
    'pgf.preamble': "\n".join([
        r"\usepackage{ctex}",   
        r"\usepackage{lmodern}",      
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{amsfonts}",
        r"\usepackage{graphicx}",
        r"\usepackage{bm}",
        r"\usepackage{gensymb}",
        r"\usepackage{microtype}", 
    ])
})


_PRELOADED_FONTS = {
    "chinese_14": None,
    "default_14": None,
}


def _is_cjk_char(char: str) -> bool:
    """Return *True* if *char* falls in any CJK Unicode block."""
    code = ord(char)
    return any(start <= code <= end for start, end in (
        (0x2E80, 0x2EFF),  # CJK Radicals Supplement
        (0x3000, 0x303F),  # CJK Symbols and Punctuation
        (0x31C0, 0x31EF),  # CJK Strokes
        (0x3200, 0x32FF),  # Enclosed CJK Letters and Months
        (0x3300, 0x33FF),  # CJK Compatibility
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xF900, 0xFAFF),  # Compatibility Ideographs
        (0xFE30, 0xFE4F),  # CJK Compatibility Forms
        (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
        (0x2CE80, 0x2EBEF),  # CJK Unified Ideographs Extension F
        (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    ))

"""格式化公式用于显示"""
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

""" LaTeX公式渲染为图像 """
@lru_cache(maxsize=1024)
def _render_latex_formula(formula: str, dpi: int = 60, fontsize: int = 4, bg_color: str = "white", bold: bool = False):
    """Render a LaTeX formula to a numpy array. Returns the image and an error message if any."""
    fig = None
    try:
        formula = f" {formula.strip()} "
        
        needs_math_mode = not (formula.startswith('$') or 
                               formula.startswith('\\begin{') or 
                               formula.endswith('$') or 
                               formula.endswith('\\]') or 
                               '\\[' in formula or 
                               formula.startswith('\\('))
        
        math_mode_required = [
            r'\begin{matrix}', r'\begin{pmatrix}', r'\begin{bmatrix}', r'\begin{Bmatrix}',
            r'\begin{vmatrix}', r'\begin{Vmatrix}', r'\begin{array}', r'\begin{cases}',
            r'\begin{aligned}',
        ]
        if any(env in formula for env in math_mode_required):
            if not formula.startswith('$') and not formula.endswith('$'):
                formula = f"${formula}$"
                needs_math_mode = False
        
        if needs_math_mode:
            formula = f"${formula}$"
        
        if bold:
            if formula.startswith('$') and formula.endswith('$'):
                formula = f"$\\bm{{{formula[1:-1]}}}$"
            else:
                formula = f"\\bm{{{formula}}}"
                
        if "\\text" in formula and not "\\usepackage{amsmath}" in rcParams['pgf.preamble']:
            formula = formula.replace("\\text{", "\\mbox{")
        
        fig_height = max(0.08, fontsize / 500) 
        fig = plt.figure(figsize=(0.08, fig_height), facecolor=bg_color)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        
        if bold:
            plt.rcParams['font.weight'] = 'bold'
        
        ax.text(0.5, 0.5, formula, fontsize=fontsize, ha='center', va='center', usetex=True)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor=bg_color)
        
        if bold:
            plt.rcParams['font.weight'] = 'normal'
        
        buf.seek(0)
        img = np.array(Image.open(buf))
        
        if img.shape[2] == 4:  # RGBA转RGB
            background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4] / 255.0
            result = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
            return result, None, formula
        return img, None, formula
        
    except Exception as e:
        error_message = f"LaTeX rendering failed: {e}"
        return None, error_message, formula
    finally:
        if fig is not None:
            plt.close(fig)

""" 获取字体 """
def get_font(size: int = 50, for_chinese: bool = False):
    """Get a font that can display math symbols and Chinese characters."""
    cache_key = f"{'chinese' if for_chinese else 'default'}_{size}"
    if cache_key in _PRELOADED_FONTS and _PRELOADED_FONTS[cache_key] is not None:
        return _PRELOADED_FONTS[cache_key]
    
    try:
        if sys.platform.startswith('win'):
            if for_chinese:
                chinese_fonts = [
                    "c:/Windows/Fonts/simsun.ttc",
                    "c:/Windows/Fonts/simhei.ttf",
                    "c:/Windows/Fonts/msyh.ttc",
                    "SimSun", "SimHei", "Microsoft YaHei", "KaiTi", "FangSong"
                ]
            else:
                chinese_fonts = [
                    "c:/Windows/Fonts/arial.ttf",
                    "c:/Windows/Fonts/times.ttf",
                    "Arial", "Times New Roman", "Courier New"
                ]
        elif sys.platform.startswith(('linux', 'darwin')):
            if for_chinese:
                chinese_fonts = [
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/System/Library/Fonts/PingFang.ttc",
                    "WenQuanYi Micro Hei", "AR PL UMing CN", "PingFang SC", "Noto Sans CJK SC"
                ]
            else:
                chinese_fonts = [
                    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "FreeSans", "Liberation Sans", "Helvetica", "Arial"
                ]
        else:
            if for_chinese:
                chinese_fonts = ["SimSun", "NotoSansSC", "Microsoft YaHei", "WenQuanYi Micro Hei"]
            else:
                chinese_fonts = ["DejaVuSans", "Arial", "Helvetica", "FreeSans"]
        
        for font_name in chinese_fonts:
            try:
                font = ImageFont.truetype(font_name, size)
                _PRELOADED_FONTS[cache_key] = font
                return font
            except (IOError, OSError):
                continue
        try:
            from matplotlib.font_manager import FontManager
            fm = FontManager()
            font_names = [f.name for f in fm.ttflist if 'cjk' in f.name.lower()]
            if font_names:
                font = ImageFont.truetype(font_names[0], size)
                _PRELOADED_FONTS[cache_key] = font
                return font
        except Exception:
            pass

        default_font = ImageFont.load_default()
        _PRELOADED_FONTS[cache_key] = default_font
        return default_font
    except Exception:
        default_font = ImageFont.load_default()
        _PRELOADED_FONTS[cache_key] = default_font
        return default_font

"""添加随机高斯噪点"""
def add_random_noise(img: np.ndarray, max_noise_intensity: int = 15) -> np.ndarray:
    img_int = img.astype(np.int16)
    noise = np.random.randint(-max_noise_intensity, max_noise_intensity + 1, img.shape, dtype=np.int16)
    noisy_img = np.clip(img_int + noise, 0, 255).astype(np.uint8)
    return noisy_img

"""随机小角度旋转"""
def random_small_rotation(img: np.ndarray, max_angle: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)  # 随机旋转角度
    # 计算旋转矩阵（以中心为原点，保持缩放）
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # 计算新尺寸
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    # 调整偏移量，确保图像居中
    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)
    # 执行旋转（白色填充空白区域）
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
    return rotated_img

"""随机组合应用1-2种增强效果（避免过度失真）"""
def apply_random_enhancements(img: np.ndarray) -> np.ndarray:    
    effects = [add_random_noise, random_small_rotation]
    selected_effects = random.sample(effects, k=random.randint(1, 2))
    enhanced_img = img.copy()
    for effect in selected_effects:
        enhanced_img = effect(enhanced_img)
    return enhanced_img


"""主处理函数"""
def process_single_task(task_info):
    img_dir, file_name, formula, output_dir, dpi, fontsize = task_info
    original_img_path = os.path.join(img_dir, file_name)  # 原图路径
    output_img_name = f"{os.path.splitext(file_name)[0]}.png"  # 输出文件名（统一PNG）
    output_img_path = os.path.join(output_dir, output_img_name)
    final_formula = formula 
    
    try:
        final_formula = _format_formula_for_display(formula)  # 格式化公式，便于渲染
        random_prob = random.random()  # 控制概率，输出原图和渲染图
        
        # 输出原图
        if random_prob >= 0.6:
            if os.path.isfile(original_img_path):
                original_img = Image.open(original_img_path).convert('RGB')
                original_img.save(output_img_path, 'PNG', dpi=(dpi, dpi))
                return (True, output_img_name, final_formula)
            else:
                return (False, None, None)
        
        # 对图片进行渲染
        render_img, error, _ = _render_latex_formula(
            final_formula, dpi=dpi, fontsize=fontsize, bg_color="white"
        )
        
        # 渲染失败 → 直接返回原图
        if error or render_img is None:
            if os.path.isfile(original_img_path):
                original_img = Image.open(original_img_path).convert('RGB')
                original_img.save(output_img_path, 'PNG', dpi=(dpi, dpi))
                return (True, output_img_name, final_formula)
            else:
                print(f"❌ 渲染失败且原图缺失：{file_name}（无输出）")
                return (False, None, None)
        
        # 应用增强
        enhanced_img = apply_random_enhancements(render_img)
        enhanced_pil = Image.fromarray(enhanced_img)
        enhanced_pil.save(output_img_path, 'PNG', dpi=(dpi, dpi))
        return (True, output_img_name, final_formula)
    
    except Exception as e:
        print(f"❌ 任务异常失败：{file_name}（错误：{str(e)[:50]}...）")
        return (False, None, None)


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="LaTeX公式渲染工具", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--img-dir", required=True, help="原图所在文件夹路径（必填）")
    parser.add_argument("-c", "--caption", required=True, help="labels.txt路径（必填，每行格式：原图文件名\\tLaTeX表达式）")
    parser.add_argument("-o", "--output", required=True, help="输出图片目录路径（必填）")
    parser.add_argument("-p", "--cpu", type=int, default=100, help="并行CPU核心数（默认100，建议≤CPU核心数）")
    parser.add_argument("-d", "--dpi", type=int, default=125, help="输出图片DPI（默认125）")
    parser.add_argument("-f", "--fontsize", type=int, default=15, help="LaTeX公式字体大小（默认15）")
    args = parser.parse_args()

    # 创建输出目录
    try:
        os.makedirs(args.output, exist_ok=True)
        print(f"📂 输出目录已准备：{os.path.abspath(args.output)}")
    except Exception as e:
        print(f"❌ 错误：创建输出目录失败 - {str(e)}")
        sys.exit(1)

    # 3. 读取caption.txt生成任务列表
    tasks = []
    with open(args.caption, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # 按第一个Tab分割（避免公式含Tab）
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"⚠️  跳过第{line_num}行：格式错误（需用Tab分隔「原图文件名」和「LaTeX公式」）")
                print(f"   错误内容：{line}")
                continue
            file_name, formula = parts
            tasks.append((args.img_dir, file_name, formula, args.output, args.dpi, args.fontsize))

    if not tasks:
        print("❌ 错误：无有效任务（caption.txt可能为空或全为错误格式）")
        sys.exit(1)
    print(f"\n🚀 任务启动：共{len(tasks)}个任务，使用{args.cpu}核并行")
    
    output_labels = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cpu) as executor:
        results = executor.map(process_single_task, tasks)
        for result in tqdm.tqdm(results, total=len(tasks), desc="处理进度"):
            has_output, img_name, formula = result
            # 仅收集有输出图片的条目
            if has_output and img_name and formula:
                output_labels.append(f"{img_name}\t{formula}")  # 按“图片名\t公式”格式存储

    output_labels_path = os.path.join(args.output, "test_labels.txt")
    if output_labels:
        try:
            with open(output_labels_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_labels))
            print(f"\n📝 已生成输出图片的labels.txt：{os.path.abspath(output_labels_path)}")
            print(f"   共包含 {len(output_labels)} 条有效图片-公式对应关系")
        except Exception as e:
            print(f"\n❌ 生成labels.txt失败：{str(e)}")
    else:
        print(f"\n⚠️  无有效输出图片，未生成labels.txt")

    print(f"\n🎉 所有任务完成！输出图片路径：{os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()