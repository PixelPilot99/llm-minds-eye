# 📊 数据集说明 | Dataset Documentation

## 目录结构 | Directory Structure

```
data/
├── raw_jpg/          # 原始图像文件 | Original image files
├── matrix/           # 矩阵格式数据 | Matrix format data
└── dsl/              # DSL格式数据 | DSL format data
    └── final/        # 最终精修数据 | Final refined data
```

## 数据版本说明 | Data Version Description

### 1. raw_jpg/ - 原始图像文件
- **内容**：手工绘制的原始图形图像文件
- **格式**：JPG格式
- **状态**：未经处理的原始素材
- **Content**: Hand-drawn original graphic image files
- **Format**: JPG format
- **Status**: Unprocessed raw materials

### 2. matrix/ - 矩阵格式数据
- **来源**：由`raw_jpg/`图像通过脚本转换生成
- **格式**：符号矩阵文本文件，32*32，部分基础结构数据为5*5
- **注意**：此目录下的描述文本**未全部修改**，可能存在不准确或不完整的描述
- **Source**: Generated from `raw_jpg/` images via script conversion
- **Format**: matrix text files，32*32, with some '基础结构' data being 5*5
- **Note**: Descriptions in this directory are **not fully revised** and may contain inaccurate or incomplete descriptions

### 3. dsl/ - DSL格式数据
- **格式**：自定义领域特定语言格式
- **特点**：采用`R<行号>:<数量><符号>...`格式压缩表示空间结构
- **注意**：此目录包含不同完成度的数据文件
- **Format**: Custom Domain-Specific Language format
- **Feature**: Uses `R<row>:<count><symbol>...` format for compressed spatial representation
- **Note**: This directory contains data files at various completion stages

### 3.1 dsl/final/ - 最终精修数据 ✨
- **内容**：44条手工校对完成的数据样本
- **状态**：描述完整、格式正确、已通过人工验证
- **用途**：建议用于模型训练和评估的基准数据集
- **Content**: 44 manually verified data samples
- **Status**: Complete descriptions, correct format, human-verified
- **Purpose**: Recommended as benchmark dataset for model training and evaluation

## 重要注意事项 | Important Notes

### 数据质量差异 | Data Quality Variations
1. **描述一致性**：仅`data/dsl/final/`目录下的数据具有完整且一致的描述
2. **格式验证**：其他目录可能存在JSON格式错误或描述缺失
3. **使用建议**：建议优先使用`final/`目录数据，其他数据需谨慎验证
1. **Description Consistency**: Only data in the `data/dsl/final/` directory has complete and consistent descriptions.
2. **Format Validation**: Other directories may contain JSON format errors or missing descriptions.
3. **Usage Recommendation**: It is recommended to prioritize data from the `final/` directory; data from other sources should be used with caution and validated.
### 技术细节 | Technical Details
- **DSL语法**：`R`表示行号，`☆`表示空白，`★`表示实体
- **转换关系**：`raw_jpg` → `matrix` → `dsl` → `dsl/final`
- **数据量**：最终精修数据集包含44个样本，覆盖纯文本、基础形状到复杂结构
- **DSL Syntax**: R indicates the row number, `☆` represents empty space , and `★` represents an entity .
- **Conversion Pipeline**: `raw_jpg` → `matrix` → `dsl` → `dsl/final`
- **Data Volume**: The final refined dataset contains 44 samples, covering pure text, basic shapes, and complex structures.

## 文件示例 | File Examples

### matrix/ 示例 (matrix/example.txt)
```
☆★☆
★★★
☆★☆
描述：这是一个十字形
```

### dsl/ 示例 (dsl/example.dsl)
```
R1:1☆1★1☆
R2:0☆3★0☆
R3:1☆1★1☆
描述：这是一个十字形图案，中心点被包围，四个方向有延伸
```

