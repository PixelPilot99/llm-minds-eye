from PIL import Image, ImageOps
import numpy as np
import json
import os
import random
from glob import glob


def image_to_symbol_matrix(image_path, symbol_pairs,debug_mode=True):
    """将图像转换为适合LLM训练的符号字符串（保留空间结构）"""
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)   #关键修复：自动校正方向
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        data = np.asarray(img.convert("L"))
        binary_matrix = (data <= 127).astype(int)

        symbol1, symbol2 = random.choice(symbol_pairs)
        symbol_matrix = np.where(binary_matrix == 1, symbol1, symbol2)

        # 新增矩阵方向验证（可选调试）
        if debug_mode:  # 可以在函数外设置debug_mode=True
            img.save("debug_resized.jpg")
            with open("debug_matrix.txt", "w") as f:
                f.write('\n'.join([''.join(row) for row in symbol_matrix]))

        return '\n'.join([''.join(row) for row in symbol_matrix])
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return None


def get_label(filename):
    base_name = os.path.splitext(filename)[0]
    if '_' in base_name:
        return base_name.split('_')[0]
    return base_name


def generate_single_json(image_path, output_dir, symbol_pairs):
    #label = get_label(os.path.basename(image_path))
    matrix = image_to_symbol_matrix(image_path, symbol_pairs)

    if matrix is not None:
        output_data = {

            "instruction": "<|im_start|>system\n这里展示了一个矩阵图像，根据结构描述画面内容，不需要输出矩阵。<|im_end|><|im_start|>user\n以自然语言描述这个物体",
            "matrix": matrix + "<|im_end|><|im_start|>assistant\n",  # 直接使用字符串格式
            "answer": '从矩阵的规律来看，☆占据了画面的四个角，应该是背景，★可能是主体内容。从分布来看，有5个比较孤立的区域从中心发散出去，到顶端越来越细小，看起来是5个尖角，外轮廓有10条边，应该是一种五角星的形状。<|im_end|><|endoftext|>'
        }

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"已生成: {output_path}")
        return True
    return False


def batch_process_images(input_dir, output_dir):
    '''
    SYMBOL_PAIRS = [
        ('@', '#'), ('$', '%'), ('&', '*'),
        ('+', '-'), ('=', '~'), ('■', '□'),
        ('●', '○'), ('★', '☆'), ('◆', '◇'),
        ('#', '0'), ('★', 'L'), ('F', '&'),
        ('+', '5'), ('=', 'A'), ('M', '□'),
        ('@', 'q'), ('$', 'T'), ('E', '*'),
    ]
    '''
    SYMBOL_PAIRS = [('★', '☆')]
    '''
    SYMBOL_PAIRS = [
        ('@', '#'), ('$', '%'), ('&', '*'),
        ('+', '-'), ('=', '~'), ('■', '□'),
        ('●', '○'), ('★', '☆'), ('◆', '◇')
    ]
    '''

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_dir, ext)))

    success_count = 0
    for img_path in image_paths:
        if generate_single_json(img_path, output_dir, SYMBOL_PAIRS):
            success_count += 1

    print(f"处理完成！成功转换 {success_count}/{len(image_paths)} 张图像")


if __name__ == "__main__":
    INPUT_DIR = "./data/raw_jpg/五角星"
    OUTPUT_DIR = "./data/matrix/五角星"
    batch_process_images(INPUT_DIR, OUTPUT_DIR)