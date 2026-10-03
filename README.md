# 企业级知识库 RAG + 多 Agent 平台

面向 AI Agent / AI 开发工程师岗位的项目：用 LangChain + LangGraph 构建一个**可评估**的企业级 RAG 系统，并升级为**多 Agent 编排**。核心思路是「先跑出基线分数，再逐项优化，用数据说话」。

## 项目亮点

- **单链 RAG + 多 Agent 双模式**：`ask` 命令是固定三步的单链 RAG，`agent` 命令是 3 个专家 Agent + 调度器的自主编排，两者可对比
- **量化驱动**：每一步优化都用 RAGAS 打分对比，形成「优化 → 评测 → 对比」闭环
- **质量闭环**：评测 Agent 会对答案自评，不通过就打回分析 Agent 重做（最多 3 次）
- **多轮对话 + 历史管理**：多轮对话带上下文（支持指代消解），历史记录支持关键词搜索、分页、回显、清空
- **真实踩坑**：完整记录了从环境搭建到评测的 6 个坑，每个都是面试素材

## 技术栈

| 环节 | 选型 | 说明 |
|------|------|------|
| LLM 生成 | DeepSeek（`deepseek-chat`） | OpenAI 兼容接口 |
| Embedding | 本地 BGE（`bge-small-zh-v1.5`） | DeepSeek 无 embedding 接口，本地跑更贴合企业私有化 |
| 向量库 | FAISS | 骨架期零部署，后续换 Qdrant |
| 混合检索 | BM25 + RRF + Rerank | 关键词精确匹配 + 语义匹配 + 精排 |
| 单链编排 | LangChain（LCEL） | retriever + prompt + llm 串成链 |
| Agent 编排 | LangGraph | 3 个专家 Agent + 调度器，状态图 + 打回重做循环 |
| Web 服务 | FastAPI | 桥接前端页面与 RAG / Agent |
| 评测 | RAGAS | 4 个指标量化检索与生成质量 |

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                       Web 前端                           │
│      流式输出 · 多轮对话 · 历史记录(搜索/分页)             │
└──────────────────────────┬──────────────────────────────┘
                           │  FastAPI (SSE)
                           ▼
┌─────────────────────────────────────────────────────────┐
│               多 Agent 编排（LangGraph）                  │
│                                                         │
│   Router ──→ 检索Agent ──→ 分析Agent ──→ 评测Agent        │
│              (自主多次检索)             │                 │
│                          不通过 → 打回重做(≤3次) ─────────┘
└─────────────┬────────────────────────────┬───────────────┘
              │                            │
              ▼                            ▼
       向量检索 FAISS                  DeepSeek LLM
              │
              ▼
       BGE Embedding
              │
              ▼
       知识库文档(4篇)

旁路评测：RAGAS 四指标（faithfulness / context_recall / context_precision / answer_relevancy）
```

## 目录结构

```
enterprise-rag/
├── config.py           # 集中配置（模型 / 切分 / 检索 / 路径）
├── main.py             # 入口：ingest / ask / agent / eval
├── server.py           # FastAPI Web 服务（桥接前端与 RAG / Agent）
├── data/
│   ├── docs/           # 中文示例文档
│   └── eval_set.json   # 15 条评测样本
├── web/                # 前端页面
│   ├── index.html
│   ├── style.css
│   └── app.js
└── src/
    ├── models.py       # LLM / Embedding 工厂
    ├── ingest.py       # 文档 → 切分 → 向量化 → 存 FAISS
    ├── retriever.py    # 向量检索 + 混合检索入口
    ├── hybrid_retriever.py  # 混合检索（BM25 + RRF + Rerank）
    ├── generator.py    # Prompt 模板
    ├── pipeline.py     # LCEL 组装 RAG 链
    ├── evaluate.py     # RAGAS 打分（含 ragas 版本冲突的 stub 绕过）
    ├── storage.py      # 历史会话存储（SQLite）
    └── agents/         # 多 Agent 模块
        ├── tools.py    # 检索工具封装
        ├── nodes.py    # 3 个 Agent 节点 + 调度器 + 状态定义
        └── graph.py    # LangGraph 图编排（含打回重做循环）
```

> 向量索引默认存到 `C:\Users\30885\.cache\rag_index`，而非项目内（原因见踩坑 #2）。

## 快速开始

### 1. 环境

**Python 3.12**（⚠️ 不要用 3.14，见踩坑 #1）：

```bash
py -3.12 -m venv .venv
# Windows cmd 激活：
.venv\Scripts\activate
```

### 2. 装依赖（国内建议加镜像）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置密钥

复制 `.env.example` 为 `.env`，填入 DeepSeek key。

### 4. 命令行运行（四条命令）

```bash
python main.py ingest                          # 文档入库（首次自动下载 BGE 模型）
python main.py ask "专业版每人每月多少钱？"      # 单链 RAG 问答
python main.py agent "帮我对比免费版和专业版"    # 多 Agent 问答
python main.py eval                            # RAGAS 评测
```

### 5. Web 界面

```bash
python server.py
```

浏览器打开 http://127.0.0.1:8000，即可在网页上提问（多 Agent 编排、流式输出、多轮对话带上下文）。侧栏「历史记录」面板自动保存历史问答，支持关键词搜索、分页、点击回显、清空。

## 多 Agent 系统

用 LangGraph 编排 3 个专家 Agent + 1 个调度器，能自主完成「检索 → 分析 → 评测 → 打回重做」闭环：

```
用户问题
   ↓
[调度器 Router] —— 判断：这个问题需要检索知识库吗？
   ├─ 需要 → [① 检索 Agent]   拆解查询、可多次检索，返回上下文
   │            ↓
   │        [② 分析 Agent]    基于上下文做对比 / 总结 / 给建议
   │            ↓
   │        [③ 评测 Agent]    自评：忠实吗？完整吗？
   │            ↓
   │        通过？ ── 否（<3次）──→ 打回 ② 重做
   └─ 不需要 → [② 分析 Agent]  （闲聊/常识直接答）
                 ↓
              通过 → 输出
```

| Agent | 角色 | 能力 |
|-------|------|------|
| 调度器 Router | 路由 | 判断问题是否需要检索知识库 |
| ① 检索 Agent | 信息获取 | 把问题拆成检索动作，内部可多次检索 |
| ② 分析 Agent | 信息加工 | 对比、归纳、生成建议 |
| ③ 评测 Agent | 质量把关 | 自评忠实度+完整性，不通过就打回 |

**为什么拆成多个 Agent**：检索、分析、评测是三种不同性质的 LLM 任务，拆成独立 Agent 后各有独立 prompt 和工具，比一个巨型 prompt 更可控，也更能体现「工具调度」和「质量闭环」。

## 评测指标

| 指标 | 含义 | 可靠性 |
|------|------|--------|
| faithfulness | 答案是否忠于检索上下文 | ✅ 可靠（单次 LLM 判断） |
| context_recall | 标准答案所需信息是否被检索到 | ✅ 可靠 |
| context_precision | 相关上下文是否排在前面 | ⚠️ 有一定噪声 |
| answer_relevancy | 答案是否切题 | ❌ 在 DeepSeek 下不可靠（见踩坑 #6） |

## 优化记录：三轮 prompt 迭代

faithfulness 从 0.64 → 1.0 的过程，体现了「忠实度 vs 相关性」的经典 trade-off：

| 版本 | faithfulness | answer_relevancy | context_precision | context_recall |
|------|-------------|-----------------|-------------------|----------------|
| v1 原始 | 0.6429 | 0.9041 | 0.8452 | 1.0000 |
| v2 强约束 | **1.0000** | 0.6923 | 0.7619 | 1.0000 |
| v3 平衡 | **1.0000** | 0.6102 | 0.8718 | 1.0000 |

**做了什么**：

- **v1 → v2**：去掉来源标注（`[来源 N]` 会被 ragas 当成"上下文没有的断言"扣分）+ 强化"逐字对应、禁止改写" → faithfulness 满分
- **v2 → v3**：放宽"逐字对应"为"允许自然转述"，加"完整回答所有方面"

**关键洞察**：

1. 来源标注会反噬 faithfulness 评测。
2. answer_relevancy 从 0.90 一路跌到 0.61，但人工抽查答案质量明明很好（如"员工手册里规定了哪几种假期？"→ 完整列出 5 种）。结论：**该指标在 DeepSeek n=1 限制下已失真，faithfulness 才是可靠的优化信号，answer_relevancy 用人工抽查兜底。**

## 混合检索 + Rerank 优化

纯向量检索对「精确词匹配」会漏（如型号、编号、专有名词），加 BM25 关键词检索 + RRF 融合 + bge-reranker 精排后：

| 指标 | 纯向量(v3) | 混合检索+Rerank | 变化 |
|------|-----------|----------------|------|
| context_precision | 0.8718 | **1.0000** | 涨到满分 |
| context_recall | 1.0000 | 1.0000 | 持平 |
| faithfulness | 1.0000 | 1.0000 | 持平 |

**核心结论**：context_precision 从 0.87 → 1.0，说明 Rerank 让最相关的 chunk 稳定排第一，这是「混合检索提升检索质量」的量化证据。

## 踩坑清单（6 个，全部亲历）

| # | 坑 | 根因 | 解法 |
|---|----|------|------|
| 1 | Python 3.14 装不上 torch/faiss | 3.14 太新，二进制包没适配 | 用 Python 3.12 |
| 2 | faiss 写索引失败 `No such file or directory` | faiss 的 C++ 底层无法读写中文路径 | 索引改存英文缓存目录 |
| 3 | `No module named langchain_community.chat_models.vertexai` | ragas 0.4.3 与新 langchain-community 版本冲突 | evaluate.py 里注册 stub 模块绕过 |
| 4 | HF 镜像不生效（仍连 huggingface.co） | HF_ENDPOINT 设置时机太晚，huggingface_hub import 时已固定 endpoint | 提前到 main.py 最顶部设置 |
| 5 | DeepSeek 无法做 embedding | DeepSeek API 只有 LLM，无 embedding 接口 | 本地 BGE 模型 |
| 6 | answer_relevancy 分数乱跳 | DeepSeek 不支持 n>1，ragas 降级成 1 个假设问题，噪声大 | 用 faithfulness 做主信号 |

## 后续优化路线

每做一步，用 `python main.py eval` 看分数变化：

1. **混合检索**：加 BM25，与向量检索做 RRF 融合（补精确词匹配）
2. **Rerank**：检索后用 `bge-reranker` 重排序
3. **Query 改写**：HyDE / 子问题分解，提升召回
4. **换更强 Embedding**：`bge-small-zh-v1.5` → `bge-m3`
5. **工程化**：FAISS → Qdrant、接入 Langfuse 做链路追踪、FastAPI 服务化
6. **数据治理**：多格式文档接入、增量更新、元数据过滤
