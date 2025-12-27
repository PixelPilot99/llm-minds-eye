import os
import json
from glob import glob


def clean_matrix_string(matrix_str):
    """清理原始矩阵中的转义符号和多余字符"""
    return matrix_str.replace('\\n', '\n').replace('\"', '').strip()


def compress_matrix(matrix_str):
    """压缩矩阵为紧凑格式，处理原始JSON的转义问题"""
    matrix_str = clean_matrix_string(matrix_str)
    rows = [row for row in matrix_str.split('\n') if row.strip()]
    compressed_rows = []

    for i, row in enumerate(rows, 1):
        compressed = f"R{i}:"
        current_char = row[0] if row else ''
        count = 0

        for char in row:
            if char == current_char:
                count += 1
            else:
                compressed += f"{count}{current_char}"
                current_char = char
                count = 1
        compressed += f"{count}{current_char}"
        compressed_rows.append(compressed)

    return '|'.join(compressed_rows)


def generate_json_from_matrix(matrix_path, output_dir):
    """从矩阵文件生成JSON"""
    try:
        with open(matrix_path, 'r', encoding='utf-8') as f:
            raw_content = json.load(f)  # 直接加载JSON

        original_matrix = raw_content["matrix"].split("<|im_end|>")[0]   #.split("<|vision_end|>")[1]
        compressed_matrix = compress_matrix(original_matrix)

        output_data = {
            "instruction": raw_content["instruction"] ,
            "matrix": f"{compressed_matrix}<|im_end|><|im_start|>assistant\n",
            "answer": raw_content["answer"]
        }

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(matrix_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"已生成: {output_path}")
        return True

    except Exception as e:
        print(f"处理 {matrix_path} 时出错: {e}")
        return False


def batch_process_matrices(input_dir, output_dir):
    """批量处理矩阵文件"""
    matrix_paths = glob(os.path.join(input_dir, '*.txt'))
    success_count = 0

    for matrix_path in matrix_paths:
        if generate_json_from_matrix(matrix_path, output_dir):
            success_count += 1

    print(f"处理完成！成功转换 {success_count}/{len(matrix_paths)} 个矩阵文件")


if __name__ == "__main__":
    INPUT_DIR = "./data/matrix/五角星"  # 原始矩阵txt存放目录
    OUTPUT_DIR = "./data/dsl/五角星"  # 输出目录
    batch_process_matrices(INPUT_DIR, OUTPUT_DIR)