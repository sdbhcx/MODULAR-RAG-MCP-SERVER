# Developer Specification: Database Schema Semantic RAG MCP

## 1. 概述 (Overview)

### 1.1 背景与目标

在 Vibe Coding（AI 辅助编程）场景中，AI 代理在编写 SQL 或业务逻辑时，常因无法获取最新、最准确的数据库表结构而产生大量幻觉（Hallucinations）。静态解析 GitLab 中的 DDL 脚本难以应对复杂的 `ALTER TABLE` 历史变更。

本项目旨在基于当前 `MODULAR-RAG-MCP-SERVER` 框架，开发一个直接连接 **开发环境数据库 (Dev DB)** 的 MCP 功能模块。通过提取真实数据库元数据、结合大模型进行业务语义增强（Metadata Enrichment），并存入向量数据库，最终向 AI 暴露精准的“意图搜索”、“结构获取”和“数据探查”工具。

### 1.2 核心特性

* **Single Source of Truth** ：直接读取 Dev DB 的 `information_schema`，确保结构的绝对正确性。
* **语义化检索 (Schema RAG)** ：通过 LLM 生成表业务意图和同义词，支持 AI 用自然语言进行跨表检索。
* **增量状态同步** ：基于 Schema Hash 比对的轻量级向量库更新机制。
* **数据探查 (Data Profiling)** ：提供只读的样本数据抽样工具，帮助 AI 秒懂字段枚举和 JSON 结构。

---

## 2. 架构设计 (Architecture Design)

系统数据流分为两条主线：**离线知识注入管道 (Ingestion Pipeline) **和  **实时在线调用管道 (MCP Tool Pipeline)** 。

### 2.1 离线知识注入管道 (Schema Ingestion & Synchronization)

1. **加载 (Loader)** ：从 Dev DB 拉取库、表、列及注释信息，序列化为结构化文本。
2. **增强 (Enrichment)** ：利用现有大语言模型接口，分析表结构生成业务摘要、潜在关联和同义词标签。
3. **哈希比对 (Hashing)** ：计算每张表的 `Hash(列结构+注释)`，比对本地缓存，决定是否跳过、新增、更新或删除。
4. **向量存储 (Vectorization/Storage)** ：将“表名+LLM增强摘要”进行向量化，将结构化详情存入 VectorDB 的 Payload 中，以 `table_name` 为主键（Document ID）。

### 2.2 实时请求调用 (MCP Interactions)

AI 在大语言模型侧发起工具调用：

* **迷茫期** ：调用意图检索工具（连接 VectorDB 进行 Hybrid Search），获取候选表名。
* **明确期** ：调用详情获取工具（直连 Dev DB 或读 VectorDB Payload），获取表的所有 DDL 细节。
* **开发期** ：调用数据探查工具（直连 Dev DB 执行只读 `LIMIT 5`），获取真实样本。

---

## 3. 模块级设计与改造计划 (Module Specifications)

基于现有项目目录结构进行功能扩展：

### 3.1 提取层加载器 (`src/libs/loader/db_schema_loader.py` -  **新增** )

* 引用的依赖：建议引入 `SQLAlchemy` 来屏蔽底层数据库类型（MySQL, PostgreSQL, etc.）差异。
* 职责：提供方法扫描库下所有业务表。
  * 读取 `information_schema.tables` (获取表级注释)
  * 读取 `information_schema.columns` (获取列名、数据类型、默认值、可空性、列级注释)
  * 组装为每张表一个 `Document` 对象。

### 3.2 元数据转换与丰富 (`src/ingestion/transform/schema_enricher.py` & [metadata_enrichment.txt](vscode-file://vscode-app/d:/software/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

* **Prompt 模板改造** ：在 `metadata_enrichment.txt` 中新增针对 DB Schema 的模板分支：
* *输入* ：DB Schema 的纯文本描述。
* *输出要求* ：生成一段 50-100 字的“核心业务意图描述”，列出“可能涉及的业务实体名词”及“同义词”，指出“可能的逻辑主外键”。
* **代码逻辑** ：接收 Loader 输出的 `Document`，调用 LLM 返回上述信息，存入 `Document.metadata`。

### 3.3 存储与向量化引擎 ([vector_store](vscode-file://vscode-app/d:/software/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) -  **功能增强** )

为了支持数据库表级的同步，这里的接口需要保证以下行为：

* 必须支持指定 `doc_id` 进行存储（强制设为 `table_name`）。
* 新增或完善 `upsert_documents()` 和 `delete_documents_by_ids()` 方法。
* 元数据要求保存：`table_name`, `schema_hash`, `last_sync_time`。

### 3.4 脚本同步任务 (`scripts/sync_db_schema.py` -  **新增** )

* 入口脚本，可以由系统的 Cron 任务或人为手动执行。
* 逻辑：
  1. 实例化 `DBSchemaLoader`。
  2. 从 `VectorStore` 读取已有的 `table_name` & `schema_hash` 映射。
  3. 全量遍历 Loader 结果：
     * Hash 相同：跳过（Skip）。
     * Hash 不同或新增：进入 `Pipeline` (Enrichment -> Embedding -> Upsert)。
     * 向量库中有但在 DB 中已不存在：执行 Delete。

---

## 4. MCP 工具接口定义 (MCP Tools Definition)

在 [tools](vscode-file://vscode-app/d:/software/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 目录下开发并注册以下工具，供大模型直接调用。

### 工具 1: `search_database_tables` (业务意图找表)

* **描述** : 根据业务需求的自然语言描述，搜索相关的数据库表。
* **输入参数** :
* `query` (string): 必填，自然语言描述（如 "查询用户的积分流水和余额"）。
* `limit` (integer): 选填，默认 3。
* **输出** : 匹配到的候选表列表，包含 `table_name` 及其 LLM 总结的 `business_intent` 描述。

### 工具 2: `get_table_schema_live` (获取精确表结构)

* **描述** : 获取特定某张表或多张表的精准结构定义，作为编写 SQL 或 ORM 实体类的参考上下文。
* **输入参数** :
* `table_names` (array of string): 必填，需要探查的表名称列表。
* **输出** : 经过格式化的表结构清单，包括列名、数据类型、注释、Primary/Foreign Key 标识。

### 工具 3: `execute_sample_query_on_dev` (真实数据抽样)

* **描述** : 执行极为基础的 SELECT 查询以探查某张表真实存储的数据格式（如 JSON 结构内含、状态枚举实存值），不可用于复杂计算。
* **输入参数** :
* `table_name` (string): 必填。
* `limit` (integer): 选填，默认 3，最大 10。
* **输出** : 包含查询结果的数据行列表（JSON 数组格式）。
* **⚠️ 安全约束 (Security Constraint)** :

1. 生成的 DB Engine / Connection 必须使用  **最低权限（Read-Only）账号** 。
2. 接收参入内部只能强拼接执行 `SELECT * FROM {table_name} LIMIT {limit}`，拒绝由 AI 自主拼接全量 SQL 以防止防注入及灾难性大表全表扫描。

---

## 5. 部署与环境依赖要求

1. **数据库权限** : 申请一个具有开发库 `SELECT` 权限的只读账号，并在此项目中配置如 [settings.yaml](vscode-file://vscode-app/d:/software/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 中新增 `db_schema_source` 块。
2. **网络互通** : 部署该 MCP 的环境应当能正常访问大模型 API、VectorDB 实例及目标业务的 Dev DB。
3. **Token 控制** : 大表（百级列数表）进行 Enrichment 时可能花费较多 Token，需在代码中设置单表最大列数阈值及截断机制保护，对于超大宽表，跳过 Enrichment，仅保留原生注释摘要。
