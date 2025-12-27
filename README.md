# llm-minds-eye
----
An exploratory data and script archive aimed at investigating whether LLMs can learn "spatial structure description" abilities **without relying on a vision encoder**. This is attempted by representing 2D silhouettes and a few three-view drawings using a **pure text/token structure** (a custom Domain-Specific Language or DSL).

一个探索性数据与脚本归档仓库：尝试用**纯文本/Token 结构**（自定义 DSL Domain-Specific Language）来表示二维剪影与少量三视图描述，用于探测/实验 LLM 在**不依赖视觉编码器**的前提下，能否学习到某种“空间结构描述”的能力。




> **Important Note (Scope and Expectations)**
>
> - This repository is an **unfinished exploration log + data（Chinese） archive**. It is not a mature benchmark, and training results are not guaranteed to be stable or reproducible.
> - I intend for it to serve as a starting point for experiments related to **data formats, tokenizers, and structured text representation**, for others to reference or build upon.
> - The code was completed through **AI-assisted programming** (I was responsible for defining requirements, designing data and iteration plans, and validation/debugging).

> 重要说明（定位与预期）
>
> - 本仓库是**未完成的探索记录 + 数据（中文）留档**，不是成熟的 benchmark，也不保证训练结果稳定可复现。
> - 我更希望它作为：**数据格式/Tokenizer/结构化文本表示**相关实验的起点，供有需要的人参考或二次研究。
> - 代码为 **AI 协同编程**完成（我负责提出需求、设计数据与迭代方案、验证与调试）。

---

## 导航
- [中文版](#中文版)
- [English Version](#english-version)


---

<a id="english-version"></a>
## English Version


## 1. Background and Motivation

Current multimodal models typically rely on a dedicated Vision Encoder to "see." But considering the core LLM network itself: can it build an understanding of 2D shapes and spatial relationships using only **structured text**?

I've observed that many LLMs become confused when dealing with spatial relationships (e.g., front/back, left/right, local structural consistency). Therefore, this project attempts to bypass visual input by directly encoding "graphics" into a **text-based DSL**, to probe the model's "blind man touching an elephant" capabilities and boundaries within a pure token-based logic.

---

## 2. Repository Content

**data：**
- `raw_jpg/` - Hand-drawn silhouettes
- `matrix/` - Intermediate products of 0/1 or symbol matrices
- `dsl/` - DSL text representations

**scripts：**
- `jpg_to_matrix.py` - jpg -> symbol matrix
- `matrix_to_dsl.py` - matrix -> DSL

**train：**
- `finetune_lora.py` - Training script (for Unsloth / Transformers)
- `load_lora.py` - Loading/Inference script

---

## 3. Core Idea: Representing "Graphic Structure" with a DSL

### 3.1 From Pixels to Symbol Matrix (Corresponds to Phase 0)
- **Data Source**: The silhouettes were initially hand-drawn by me in Photoshop to ensure controllable shapes, proportions, and spatial relationships.
- **Conversion**: A script converts `.png` files into symbol matrices (e.g., foreground=1, background=0, or using two distinct symbols).

### 3.2 Why a DSL?
Initially, I used long, repetitive strings like `☆☆☆☆☆★★★★...` to represent rows. I suspected that:
- Long repetitive sequences are unfriendly to tokenizers.
- Structural information might be destroyed after tokenization.
- The model might learn "format noise" rather than "spatial logic."

Therefore, I designed a more "linguistic" and compressed row-level representation, similar to the idea of Run-Length Encoding (RLE).

### 3.3 DSL Format (Example)
Each row is represented by "row number + several segments (count + symbol)":

- Format (Schematic):
  - `R<row_number>:<count><symbol><count><symbol>...`
- Example (Single Row):
  - `R17:2☆1★10☆6★10☆1★2☆`
- Multiple rows can be connected with a separator (Schematic):
  - `R17:...|R18:...|R19:...`

> Note: The symbols themselves can be `0/1`, `☆/★`, or others, as long as they are consistent throughout the dataset.

---

## 4. Dataset Iteration History (Preserving "Failed Paths" for Reference)

> Note: I have not received formal training in this area, so the data creation process was rather "native/iterative." I chose to preserve data from different stages to help others understand the evolution and pitfalls.

**A Note on Data Format**:
The data is stored in the `data/` directory, with each sample in a `.txt` file.
Although the extension is `.txt`, the **content is actually in JSON format**.
The fields include Qwen's training template tags (e.g., `<|im_start|>`), which was done to replicate my experimental environment at the time.
**You will need to write your own scripts to read and clean the data**.

### Phase 0: Data Generation and Preprocessing (Artistic Foundation)
- Hand-drew silhouettes (jpg).
- Batch-processed with scripts: jpg → 0/1-like matrices / symbol matrices.

### Phase 1: Matrix Prototype (Accumulated ~250+ early samples)
1. Initially used `32×32` matrices to represent silhouettes with **two random symbols** to prevent the model from overfitting to specific symbols.
2. The format was "what you see is what you get" but not standardized, and instructions were simple (e.g., `[Identify the shape]`).
3. Result: The model's output was chaotic. Too many variables (random symbols + unstructured format) made the learning target unstable.
4. To reduce variables, I later fixed the symbols to `☆/★`, but the results were still unsatisfactory.

### Phase 1.5: Instruction Shift—From "Recognition" to "Description" (Important Update)
I later realized that asking a small model to directly "recognize/perceive" was too difficult. I redefined the task to be a generation task, which is more aligned with an LLM's strengths:

- No longer asking, "What shape is this?"
- Instead: **Given an object's name/concept** (e.g., "arrow," "chair," "symmetrical shape"), the model is required to **describe its structure using the DSL or a structured format**.
- During training, a loss mask (e.g., `ignore_index=-100`) was applied to the input side (object name/prompt) to force the model to primarily learn the conditional generation mapping:
  - `concept/name` → `structured description`

This step transformed an uncontrollable "perception" problem into a more trainable "structured text generation" problem.

### Phase 2: Core Representation Iteration, Introducing DSL
- Goal: To mitigate the tokenizer's destructive effect on long, repetitive strings and make the data more "language-like" rather than a "pixel wall."
- Method: Encoded each row as `R<row_number>:<count><symbol>...` and connected rows with a separator.
- Result: The data became more structured, and the model could at least produce "decently formatted" output more consistently.

### Phase 3: Increasing Conceptual Complexity
- Added abstract spatial concepts to the descriptions:
  - Point symmetry, axial symmetry, translation, rotation, etc.
- Added a small number of "three-view drawings" (front/top/side):
  - Extremely labor-intensive.
  - The accuracy of descriptions generated by commercial LLMs was often below 50%.
  - Required extensive **manual correction and rewriting**.

### Phase 4: Final "Standard" Subset (44 samples)
- After studying how existing datasets emphasize "format consistency," I meticulously refined each sample to unify the descriptive style and format.
- This subset was used to **get the training pipeline running**:
  - After training, the small model mostly stopped producing gibberish.
  - It could generate sentences with a similar format.
  - However, it still failed significantly in **precise counting and rigorous structure** (likely due to the small dataset size, overfitting, tokenizer representation, etc.).

---

## 5. Training and Inference (Status: "Runnable," not "Reproducible")

### 5.1 Environment
- I conducted most of the work in a **Windows 10** environment, which introduced additional CUDA and dependency compatibility issues.
- Due to platform limitations, certain Linux optimization components (like Triton) may not be available.

- Test Environment (My Config)
  - OS: Windows 10
  - Python: 3.12
  - PyTorch: 2.4.1 + CUDA 12.1 (cu121)
  - torchvision: 0.19.1
  - torchaudio: 2.4.1
  - transformers: 4.51.3
  - xformers: 0.0.28.post1
  - unsloth: 2025.4.5
  - triton: 3.3.0 (Effectively inactive on Windows 10)

> Note: This repository focuses on dataset archiving and experimental procedures. Training results may vary due to differences in environment and library versions.

### 5.2 Training Method (Illustrative)
- Data is structured text (prompt/answer) in JSON format within .txt files for easy manual editing.
- Data concatenation is handled directly within `finetune_lora.py`.
- Used LoRA/PEFT for fine-tuning a small model (script in `train/finetune_lora.py`).
- Model: Qwen2.5-1.5B   
- Model: local snapshot `./models/Qwen2.5-1.5B` (Qwen2.5 1.5B series)
- Loading: `load_in_4bit=True` (4-bit quantized; QLoRA-style)
- Context: `max_seq_length=3000`
- RoPE: dynamic scaling (`factor=2.0`) for position extrapolation
- LoRA config:
  - r=4, lora_alpha=8, lora_dropout=0
  - target_modules = `q_proj`, `k_proj`, `v_proj`
  - gradient checkpointing enabled
- The model was chosen primarily to validate the workflow, not because it is the optimal choice.

### 5.3 Unsloth Local Model Loading Issue (Workaround, possibly version-specific)
I encountered an issue where Unsloth repeatedly tried to download a model online and failed to read a local small model (even if it was in the mapper list). I eventually located a flag in the source code:

- Location (Illustrative):loader.py: `class FastLanguageModel(FastLlamaModel):`
- From: `use_exact_model_name = False`（line96）
- To: `use_exact_model_name = True`

> Disclaimer: This was a local workaround for my specific environment and version. It may not apply to all versions. 

### 5.4 LoRA Adapter Loading Compatibility
- I experienced instability when loading LoRA adapters with Unsloth.
- I subsequently switched to the native Hugging Face Transformers method for inference (see `train/load_lora.py`).

---

## 6. Qualitative Observations (Not statistical, not a benchmark)

> The following observations are primarily from interactions with various commercial LLMs during data creation and correction. They are not systematic statistics and may be influenced by prompts, randomness, and model versions.

1. **Difference in Row vs. Column Structure Perception**  
   Many models find it easier to "see" structural changes along rows (e.g., a peak at the top) but are less sensitive to incremental/decremental changes along columns (e.g., a horizontal point).
2. **Protrusions are Easier to Identify than Concavities**  
   Protruding features are often described more accurately, while concave or indented structures are more likely to be ignored or described incorrectly.
3. **Providing the Object's Name First Significantly Improves Descriptions**  
   When told what the object is beforehand (name/concept), the model can often match more structural details. However, it often exhibits a pattern where the "upper part is more accurate, and the lower part starts to drift" (e.g., confusing left/right, front/back).
4. **Noticeable Differences in Spatial Sense Among Models**  
   There were palpable differences in capability, but no rigorous evaluation was performed, so probabilistic factors cannot be ruled out.

---

## 7. Known Limitations & Next Steps (Roadmap)
- **Small Data Scale**: The final refined subset contains only 44 samples, mainly for validating the pipeline.
- **Potential Tokenizer Conflict**: The representation may still conflict with the tokenizer, potentially destroying structural information during tokenization.
- **Training Bias towards "Format Learning"**: The small model may have learned to output the correct format but not the strict logic/counting.
- **Lack of Systematic Evaluation**: There are currently no unified metrics or control experiments.

Possible next steps (if this work is continued):
- Further constrain the DSL into a set of tokens that are easier to tokenize (e.g., fixed alphanumeric characters and separators).
- Augment with synthetic data (procedurally generate symmetrical/rotated/translated samples) to increase scale.
- Introduce an automatic validator (to reversibly decode the DSL back into a matrix and check for consistency).
- Conduct minimal control experiments: different tokenizers / different representations / different model sizes.

---

## 8. Usage and Contribution
- Feel free to submit PRs/Issues regarding the data format, validation scripts, or training process.
- If you use this data or the DSL concept, please consider linking back to this repository in your project (or citing it in your paper/report).

---

## 9. License
- MIT 

---

## 10. Acknowledgements
Thanks to the various LLM tools that assisted in the data format iteration, debugging, and script-building process. Thanks also to the open-source community for their contributions to toolchains like Transformers, PEFT, and Unsloth.






<a id="中文版"></a>
## 中文版

[↑ Back to Top 返回顶部](#llm-minds-eye)

## 1. 背景与动机

当前多模态模型通常依赖专门的视觉编码器（Vision Encoder）来“看”。但如果只看 LLM 核心网络本身：它能否仅通过**结构化文本**建立对二维形状/空间关系的理解？

我观察到很多 LLM 在涉及空间关系时会出现混乱（例如前后、左右、局部结构一致性等）。因此本项目尝试绕开视觉输入，直接将“图形”编码为**文本 DSL**，探测模型在纯 token 逻辑下的“盲人摸象”能力边界。

---


## 2. 仓库内容

**data：**

- `raw_jpg/` - 手绘剪影
- `matrix/` - 0/1 或符号矩阵的中间产物
- `dsl/` - DSL 文本表示

**scripts：**

- `jpg_to_matrix.py` - jpg -> 符号矩阵
- `matrix_to_dsl.py` - 矩阵 -> DSL

**train：**

- `finetune_lora.py` - 训练脚本（Unsloth / Transformers 任一）
- `load_lora.py` - 加载/推理脚本

---

## 3. 核心想法：用 DSL 表示“图形结构”

### 3.1 从像素到符号矩阵（Phase 0 对应）
- **数据来源**：剪影图最初由我在 Photoshop 手工绘制，保证形状、比例、空间关系可控。
- **转换**：用脚本将 `.jpg` 转为符号矩阵（例如前景=1、背景=0，或用两种符号表示）。

### 3.2 为什么需要 DSL？
早期我使用类似 `☆☆☆☆☆★★★★...` 这类长重复字符串表达行内容，怀疑：
- 长重复序列对 tokenizer 不友好；
- 结构信息在分词后可能被破坏；
- 模型更容易学“格式噪声”，而不是学“空间逻辑”。

因此我设计了一个更“语言化”、更压缩的行级表示（类似行程编码/RLE 的思路）。

### 3.3 DSL 格式（示例）
用“行号 + 若干段（数量+符号）”表达每一行：

- 形式（示意）：
  - `R<row_number>:<count><symbol><count><symbol>...`
- 示例（单行）：
  - `R17:2☆1★10☆6★10☆1★2☆`
- 多行可用分隔符连接（示意）：
  - `R17:...|R18:...|R19:...`

> 备注：符号本身可以是 `0/1`、`☆/★` 或其他，只要全数据集统一即可。

---

## 4. 数据集迭代记录（保留“失败路径”，供参考）

> 说明：我未受过该方向的系统训练，因此数据形成经历比较“原生/迭代式”。我选择把不同阶段的数据也保留，方便他人理解演化过程与踩坑点。

**关于数据格式的说明**：
数据存放在 data/ 目录下，每个样本是一个 .txt 文件。
虽然后缀是 txt，但**内容其实是 JSON 格式**。
字段包含了 Qwen 的训练模板（<|im_start|>等），这是为了还原我当时的实验环境。
**请自行编写脚本读取和清洗**。

### Phase 0：数据生成与预处理（艺术基础）
- 手绘剪影（jpg）
- 脚本批处理：jpg → 类似0/1 矩阵/符号矩阵

### Phase 1：矩阵原型（约 250+ 的早期样本累计）
1. 最初用 `32×32` 矩阵表示剪影，并使用**随机的两种符号**，希望模型不要过拟合单一符号。
2. 格式“所见即所得”但不规范，指令也很简单（例如：`[识别图形]`）。
3. 结果：模型输出非常混乱，变量过多（随机符号 + 非结构化格式），学习目标不稳定。
4. 为减少变量，后续暂时固定为 `☆/★` 两符号，但效果仍不理想。

### Phase 1.5：指令转向——从“识别”到“描述”（重要补充）
我后来意识到“让小模型直接识别/感知”过难，于是将任务重定义为更贴近 LLM 强项的生成任务：

- 不再问“这是什么形状？”
- 改为：**给定物体名称/概念**（如“箭头/椅子/对称图形”等），要求模型**用 DSL 或结构化格式描述其结构**。
- 训练时对输入侧（物体名称/提示）进行 loss mask（例如 `ignore_index=-100`），迫使模型主要学习：
  - `概念/名称` → `结构化描述` 的条件生成映射

这一步把不可控的“感知”问题，转为更可训练的“结构化文本生成”问题。

### Phase 2：核心表示法迭代，引入 DSL（Domain-Specific Language）
- 目的：降低 tokenizer 对长重复串的破坏；让数据更像“语言”而非“像素墙”。
- 方案：按行编码 `R<行号>:<数量><符号>...`，并用分隔符连接多行。
- 结果：数据变得更规整，模型至少能更稳定地产生“像样的格式”。

### Phase 3：概念复杂度增加
- 描述中加入抽象空间概念：
  - 中心对称、轴对称、平移、旋转等
- 加入少量“三视图”（正/俯/侧）：
  - 工作量非常大
  - 市面 LLM 辅助生成的描述准确率常低于 50%
  - 需要大量**纯手工纠正与重写**

### Phase 4：最终“标准”子集（44 条）
- 在参考现成数据集对“格式一致性”的重视后，我对样本进行逐条精修、统一描述风格与格式。
- 用于**跑通训练管线**：
  - 训练后小模型基本不再输出乱码
  - 能生成相似格式的语句
  - 但在**精确计数/严谨结构**上仍明显失败（可能与数据量小、过拟合、tokenizer 表示等有关）

---

## 5. 训练与推理（状态：仅保证“可跑通”，不保证稳定复现）

### 5.1 环境说明
- 我主要在 **Windows 10** 环境折腾完成（这会带来额外的 CUDA/依赖兼容问题）。
- 由于平台限制，某些 Linux 优化组件（如 Triton 等）可能不可用。

- 测试环境 (我的配置)
  - OS: Windows 10
  - Python: 3.12
  - PyTorch: 2.4.1 + CUDA 12.1 (cu121)
  - torchvision: 0.19.1
  - torchaudio: 2.4.1
  - transformers: 4.51.3
  - xformers: 0.0.28.post1
  - unsloth: 2025.4.5
  - triton: 3.3.0 (Windows 10中实际无效)

> Note: 本仓库专注于数据集归档与流程实验。训练结果可能因环境与版本差异而有所不同。


### 5.2 训练方式（示意）
- 数据为结构化文本（prompt/answer），json格式的txt，便于手动修改。
- 数据拼接直接在finetune_lora中完成了
- 使用 LoRA/PEFT 进行小模型微调（脚本见 `train/finetune_lora.py`）
- 模型：Qwen2.5-1.5B   
- Model: local snapshot `./models/Qwen2.5-1.5B` (Qwen2.5 1.5B series)
- Loading: `load_in_4bit=True` (4-bit quantized; QLoRA-style)
- Context: `max_seq_length=3000`
- RoPE: dynamic scaling (`factor=2.0`) for position extrapolation
- LoRA config:
  - r=4, lora_alpha=8, lora_dropout=0
  - target_modules = `q_proj`, `k_proj`, `v_proj`
  - gradient checkpointing enabled
- 模型选择主要是为了运行流程；并不代表是最佳选择。


### 5.3 Unsloth 本地模型加载问题（workaround，可能版本相关）
我遇到过一个情况：Unsloth 反复尝试在线下载模型，无法读取本地小模型（即使模型在 mapper 列表中）。最终在源码中定位到一处 flag：

- 位置(示意)：loader.py下：`class FastLanguageModel(FastLlamaModel):`
- 将：`use_exact_model_name = False`（第96行）
- 改为：`use_exact_model_name = True`

> 免责声明：这是我在当时环境/版本下的本地 workaround，未必适用于所有版本。

### 5.4 加载 LoRA 适配器的兼容性
- 我遇到过 Unsloth 加载 LoRA adapter 不稳定的问题
- 后续改用 Hugging Face Transformers 原生方式加载完成推理（见 `train/load_lora.py`）

---

## 6. 定性观察（非统计、非 benchmark）

> 以下观察主要来自数据制作/校对过程中与多种商业 LLM 的互动，以及对其输出的人工纠偏。没有系统性统计，可能受提示词、随机性与模型版本影响。

1. **行/列结构感知差异**  
   很多模型更容易“看见”行方向的结构变化（例如上尖角），但对列方向的递增/递减结构更不敏感（例如横向尖角）。
2. **凸出比凹陷更易识别**  
   突出特征往往更容易被描述正确，凹陷/内凹结构更容易被忽略或写错。
3. **先给物体名称会显著改善描述**  
   预先告知“这是什么”（名称/概念）后，模型往往能对上更多结构；但常出现“上半部分较准确、下半部分开始跑偏”的现象（左右/前后混淆等）。
4. **不同模型的空间感能力差异明显**  
   有体感差距，但未做严格评测，不能排除概率因素。

---

## 7. 已知限制与下一步（Roadmap）
- 数据规模小：最终精修子集仅 44 条，主要用于跑通流程
- 表示法仍可能与 tokenizer 冲突：结构信息可能在分词后被破坏
- 训练目标偏“格式学习”：小模型可能学会输出格式，但学不会严格逻辑/计数
- 缺少系统评测：目前没有统一指标与对照实验

可能的下一步（如果后续继续）：
- 将 DSL 进一步约束为更易分词的 token 集合（例如固定字母数字与分隔符）
- 增加合成数据（程序生成对称/旋转/平移样本），扩大规模
- 引入自动校验器（对 DSL 可逆解码回矩阵，并做一致性检查）
- 做最小的对照实验：不同 tokenizer / 不同表示法 / 不同模型尺寸

---

## 8. 使用与贡献
- 欢迎对数据格式、校验脚本、训练流程提出 PR/Issue
- 如果你用到了本数据或 DSL 思路，建议在你的项目中链接回本仓库（或在论文/报告中引用）

---

## 9. 许可（License）
- MIT 

---

## 10. 致谢
感谢各类 LLM 工具在数据格式迭代、调试与脚本搭建过程中的协助；同时也感谢开源社区在 Transformers/PEFT/Unsloth 等工具链上的贡献。

[↑ Back to Top 返回顶部](#llm-minds-eye)

