# 面向特定领域的基于思维链蒸馏的大模型指令微调实践
**Practice on Instruction Fine-Tuning of Large Language Models via Chain-of-Thought Distillation in Vertical Domains**

> **课程作业提交**：大模型全生命周期实践
> **课题方向**：结合思维链蒸馏（CoT）、检索增强（RAG）与规则累积的定密技术研究

## ⚠️ 重要声明 (Disclaimer)

**请注意：**
本仓库中的代码和数据**仅作为课程作业的逻辑展示与实现证明**，无法直接运行。
* **数据**：`data.json` 中仅包含少量样例数据，用于展示思维链（CoT）数据的格式结构。
* **模型文件**：`Qwen3-8B/merged-model` 目录下的文件夹仅包含微调模型的部分权重文件（文件体积过大）。

## 📂 项目结构说明 (Repository Structure)

本项目实现了从数据构建、指令微调、模型合并到推理评估的完整 LLM 生命周期流程。以下是主要文件及目录的详细说明：

### 1. 框架与配置 (Framework & Config)
* **`llama-factory/`**
    * 项目核心训练框架，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 构建。
* **`train.yaml`**
    * **作用**：指令微调（SFT）阶段的参数配置文件。
    * **内容**：定义了针对 Qwen3 模型的训练超参数，包括学习率（`learning_rate`）、Epoch 数、LoRA 目标模块（`lora_target: all`）以及 DeepSpeed 配置。
* **`merge.yaml`**
    * **作用**：模型权重合并与导出的配置文件。
    * **内容**：配置了 `export_device: cpu` 和 `export_size`，用于将训练好的 LoRA 适配器合并回基座模型，以便于后续的 vLLM 部署。

### 2. 数据资产 (Data)
* **`data.json`**
    * **作用**：思维链蒸馏数据的样例展示。
    * **格式**：标准的 Alpaca 格式（`instruction`, `input`, `output`）。
    * **特点**：`output` 字段中包含了由教师模型（DeepSeek-R1）生成的 `<think>` 标签，展示了“推理过程+最终结论”的数据结构。

### 3. 核心逻辑实现 (Core Implementation)
* **`CombineChain.py`**
    * **作用**：报告中提到的“组合方案（Combined Scheme）”的具体代码实现。
    * **技术栈**：基于 **LangChain** 框架编写。
    * **逻辑**：实现了 RAG（检索增强）与 规则累积（Rule Accumulation）的串联逻辑。它负责检索相似样例、动态注入校验规则，并构建最终 Prompt 调用 vLLM 接口。
* **`test.py`**
    * **作用**：实验整体测试脚本。
    * **逻辑**：遍历测试集，调用微调后的模型（或 `CombineChain` 流程），将输出结果与 Ground Truth 进行比对，并计算精确率、召回率、F1 值及指令遵循错误率。

### 4. 模型产物 (Model Artifacts)
* **`Qwen3-8B/lora/sft/`**
    * 微调过程中保存 LoRA 适配器权重（Adapter Weights）和 Checkpoints 的输出路径。
* **`Qwen3-8B/merged-model/`**
    * 合并 LoRA 权重后的完整模型文件路径。

## 🛠️ 技术流程 (Workflow)

本项目遵循标准的大模型落地生命周期：

1.  **数据准备**：利用教师模型进行 CoT 蒸馏 $\rightarrow$ 生成 `data.json`
2.  **模型微调**：使用 LLaMA-Factory 加载 `train.yaml` $\rightarrow$ 训练 LoRA 适配器
3.  **模型部署**：使用 `merge.yaml` 导出权重 $\rightarrow$ 启动 vLLM API Server
4.  **应用推理**：运行 `CombineChain.py` $\rightarrow$ 结合 RAG 与规则库进行推理
5.  **效果评估**：运行 `test.py` $\rightarrow$ 输出实验指标表格

## 📧 联系方式

如果您对报告中的实验细节或代码逻辑有任何疑问，欢迎随时联系。

---
*本仓库属于课程期末作业提交材料的一部分。*