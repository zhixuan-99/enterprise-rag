# 简历项目描述

> 以下内容可直接复制到简历「项目经历」栏。

## 精简版（简历用，3-4 行）

**企业级知识库 RAG + 多 Agent 问答平台** · Python / LangChain / LangGraph / FastAPI

- 用 LangGraph 编排「调度器 + 检索/分析/评测」三个专家 Agent，实现自主检索、质量自评、不通过打回重做的闭环
- 搭建 RAGAS 四指标评测体系，通过三轮 prompt 迭代将忠实度 faithfulness 从 0.64 优化至 1.0
- 落地 SSE 流式输出、多轮对话（指代消解）、历史记录搜索/分页等完整工程能力

## 完整版（面试讲项目用）

**项目**：企业级知识库 RAG + 多 Agent 问答平台
**技术栈**：Python、LangChain（LCEL）、LangGraph、FAISS、BGE、DeepSeek、FastAPI、RAGAS、SQLite

**做了什么**：构建了一个可量化评测的知识库问答系统，从单链 RAG 一路升级到多 Agent 编排，覆盖「检索 → 生成 → 评测 → 打回重做」完整闭环。

**技术亮点**：
1. **多 Agent 编排**：LangGraph 状态图编排「调度器 Router + 检索/分析/评测」三个专家 Agent，检索 Agent 可自主多次检索并做指代消解，评测 Agent 对答案自评、不通过打回重做（最多 3 次）。
2. **量化驱动优化**：搭建 RAGAS 评测闭环，通过三轮 prompt 迭代把 faithfulness 从 0.64 提到 1.0；并识别出「answer_relevancy 在 DeepSeek n=1 限制下失真」这一评测可靠性边界，用 faithfulness 做主信号。
3. **工程化落地**：FastAPI + SSE 流式输出、多轮对话带上下文、历史记录（SQLite + 搜索 + 分页）、精致 Web 前端。

**量化成果**：faithfulness 0.64 → 1.0，context_recall 1.0。

**踩过的坑（面试素材）**：Python 3.14 太新装不上 torch/faiss → 换 3.12；faiss 无法读写中文路径（C++ 编码问题）；ragas 与 langchain 版本冲突（stub 绕过）；HF 镜像设置时机；DeepSeek 无 embedding 接口、不支持 n>1。

## 关键词（简历筛选用）

Agent · RAG · Python · LangChain · LangGraph · FAISS · FastAPI · RAGAS · 流式输出 · 多轮对话 · 评测
