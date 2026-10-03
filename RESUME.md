# 简历项目描述

> 以下内容可直接复制到简历「项目经历」栏。

## 精简版（简历用，3-4 行）

**企业级知识库 RAG + 多 Agent 问答平台** · Python / LangChain / LangGraph / FastAPI

- 用 LangGraph 编排「调度器 + 检索/分析/评测」三专家 Agent，实现自主检索、质量自评、打回重做的闭环
- 实现 BM25 + 向量混合检索 + Rerank 精排，context_precision 提升至满分 1.0；通过 prompt 迭代将 faithfulness 优化至 1.0
- 搭建 RAGAS 评测闭环，落地 SSE 流式输出、多轮对话（指代消解）、历史记录搜索/分页

## 完整版（面试讲项目用）

**项目**：企业级知识库 RAG + 多 Agent 问答平台
**技术栈**：Python、LangChain（LCEL）、LangGraph、FAISS、BM25/RRF/Rerank、BGE、DeepSeek、FastAPI、RAGAS、SQLite

**做了什么**：构建了一个可量化评测的知识库问答系统，从单链 RAG 升级到多 Agent 编排，覆盖「混合检索 → 生成 → 评测 → 打回重做」完整闭环。

**技术亮点**：

1. **多 Agent 编排**：LangGraph 状态图编排「调度器 + 检索/分析/评测」三个专家 Agent，检索 Agent 可自主多次检索并做指代消解，评测 Agent 自评答案、不通过打回重做（最多 3 次）。
2. **混合检索 + Rerank**：针对纯向量检索对精确词匹配的不足，实现 BM25 + 向量 + RRF 融合 + bge-reranker 精排，context_precision 从 0.87 提升至 1.0。
3. **量化评测闭环**：搭建 RAGAS 四指标评测，通过三轮 prompt 迭代将 faithfulness 从 0.64 优化至 1.0，并识别出 answer_relevancy 在 DeepSeek n=1 限制下失真的评测可靠性边界。
4. **工程化落地**：FastAPI + SSE 流式输出、多轮对话带上下文、历史记录（SQLite + 搜索 + 分页）、Web 前端。

**量化成果**：faithfulness 0.64 → 1.0；context_precision 0.87 → 1.0；context_recall 1.0。

**踩过的坑**：Python 3.14 生态未适配 → 换 3.12；faiss 中文路径崩溃（C++ 编码）；ragas 与 langchain 版本冲突（stub 绕过）；HF 镜像时机；DeepSeek 无 embedding / 不支持 n>1。

## 关键词（简历筛选用）

Agent · RAG · Python · LangChain · LangGraph · 混合检索 · Rerank · FastAPI · RAGAS · 流式输出 · 多轮对话
