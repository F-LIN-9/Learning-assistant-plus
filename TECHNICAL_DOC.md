# 多智能体学习系统 - 技术文档

## 一、项目概述

### 1.1 项目背景
传统学习系统存在以下痛点：
- **单一功能**：只能提供单一的学习服务（如仅答疑或仅资源生成）
- **缺乏个性化**：无法根据学生画像提供定制化学习方案
- **AI 幻觉问题**：大语言模型生成的内容可能存在事实性错误
- **学习路径混乱**：学生不知道应该按什么顺序学习知识点

### 1.2 解决方案
本项目基于**多智能体协同架构**，构建了一个完整的学习生态系统：
- **14 个智能 Agent**：覆盖学习全流程
- **防幻觉机制**：6 维度检测 + 交叉验证
- **知识图谱**：自动构建学习路径
- **多模态支持**：文本/图片/视频/OCR

### 1.3 技术栈
| 层级 | 技术选型 |
|------|----------|
| 前端 | HTML5 + CSS3 + JavaScript（原生） |
| 后端 | Python 3.11 + Flask 2.3+ |
| 数据库 | SQLite 3 |
| AI 服务 | DeepSeek / 讯飞星火 / SiliconFlow / Replicate / 通义万相 |
| 部署 | Render 云平台 + GitHub |

---

## 二、系统架构

### 2.1 整体架构
```
用户端 → Flask 后端 → Orchestrator → 14 个 Agent → AI 服务
                    ↓
              知识图谱 → 学习路径
                    ↓
              消息总线 → 防幻觉检测
```

### 2.2 核心模块

#### 2.2.1 Orchestrator 调度器
- **职责**：任务分发、状态管理、SSE 事件流推送
- **特点**：支持异步任务调度，实时反馈进度

#### 2.2.2 14 个智能 Agent
| Agent 名称 | 职责 | 技术实现 |
|-----------|------|----------|
| ProfileAgent | 学生画像构建 | 动态问卷 + 画像更新 |
| TutorAgent | 智能答疑 | DeepSeek + 流式输出 |
| ResourceAgent | 资源生成 | Orchestrator 调度 |
| PathAgent | 学习路径 | 知识图谱 + 拓扑排序 |
| EvalAgent | 学习评估 | 5 维度评估模型 |
| OCRAgent | 拍照搜题 | 讯飞 OCR + AI 解答 |
| ImageAgent | 图片生成 | Replicate SDXL / SiliconFlow |
| VideoAgent | 视频生成 | 通义万相 / SiliconFlow |
| AntiHallucination | 防幻觉检测 | 6 维度检测 |
| KnowledgeGraph | 知识图谱 | 有向图 + 依赖分析 |
| MessageBus | 消息总线 | Agent 协同可视化 |
| TrackAgent | 行为跟踪 | 学习数据记录 |
| StrategyAgent | 策略推荐 | 预测模型 + 推荐算法 |
| MultimodalAgent | 多模态一致性 | 跨模态验证 |

---

## 三、核心技术

### 3.1 防幻觉机制

#### 3.1.1 6 维度检测
```python
HALLUCINATION_PATTERNS = [
    r"据.*研究.*表明",      # 虚假引用
    r"可能.*也许.*大概",    # 不确定性表达
    r"尚未.*证实",          # 未验证信息
    r"目前.*没有.*证据",    # 否定性陈述
]
```

| 检测维度 | 检测方法 | 处理方式 |
|----------|----------|----------|
| 事实性 | 关键词匹配 + 交叉验证 | 标记警告 |
| 逻辑性 | 前后矛盾检测 | 重新生成 |
| 一致性 | 多 Agent 互相校验 | 取共识结果 |
| 时效性 | 时间敏感词检测 | 提示用户核实 |
| 安全性 | 敏感词过滤 | 内容拦截 |
| 完整性 | 重复内容检测 | 自动过滤 |

#### 3.1.2 交叉验证流程
```
AI 生成内容 → 6 维度检测 → 交叉验证 → 通过？ → 输出结果
                                      ↓ 否
                                   标记警告
```

### 3.2 知识图谱构建

#### 3.2.1 知识点提取
```python
KNOWLEDGE_GRAPH = {
    "勾股定理": {
        "difficulty": 3,
        "prerequisites": ["平方根", "三角形"],
        "category": "数学"
    },
    "牛顿第二定律": {
        "difficulty": 4,
        "prerequisites": ["力的概念", "加速度"],
        "category": "物理"
    }
}
```

#### 3.2.2 拓扑排序算法
```python
def generate_learning_path(topic, profile):
    # 1. 获取知识点依赖
    prereqs = get_prerequisites(topic)
    
    # 2. 根据学习模式过滤
    if profile.get("learning_mode") == "quick":
        prereqs = [p for p in prereqs if difficulty[p] <= 3]
    
    # 3. 拓扑排序
    path = topological_sort(prereqs)
    
    # 4. 返回学习路径
    return path
```

### 3.3 多智能体协同

#### 3.3.1 消息总线
```python
class MessageBus:
    def __init__(self):
        self.messages = []
    
    def publish(self, agent_name, message):
        self.messages.append({
            "from": agent_name,
            "message": message,
            "timestamp": time.time()
        })
    
    def subscribe(self, agent_name):
        return [m for m in self.messages if m["from"] == agent_name]
```

#### 3.3.2 Agent 协同关系
```
ProfileAgent → TrackAgent → StrategyAgent → ProfileAgent (闭环)
TutorAgent → AntiHallucination → MessageBus
ResourceAgent → AntiHallucination → MessageBus
ResourceAgent → MultimodalAgent → MessageBus
```

---

## 四、工程优化

### 4.1 API 重试机制
```python
def retry_request(func, max_retries=3, backoff=1.0):
    """带重试和退避的 HTTP 请求包装器"""
    for attempt in range(max_retries):
        try:
            result = func()
            if result.status_code in [200, 201, 202]:
                return result
            if result.status_code in [429, 500, 502, 503]:
                time.sleep(backoff * (2 ** attempt))
                continue
            return result
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(backoff * (2 ** attempt))
    raise Exception("Request failed after retries")
```

### 4.2 输入验证
```python
def validate_input(text, max_length=1000):
    """验证用户输入"""
    if not text:
        return False, "输入不能为空"
    if len(text) > max_length:
        return False, f"输入过长，请控制在{max_length}字以内"
    dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onload=']
    if any(p in text.lower() for p in dangerous_patterns):
        return False, "输入包含非法字符"
    return True, ""
```

### 4.3 数据库索引优化
```sql
CREATE INDEX idx_users_data_username ON users_data(username);
CREATE INDEX idx_users_data_type ON users_data(data_type);
CREATE INDEX idx_users_data_created ON users_data(created_at);
CREATE INDEX idx_resources_username ON resources(username);
CREATE INDEX idx_resources_created ON resources(created_at);
```

### 4.4 环境变量管理
```python
def _inject_env_vars(config):
    """强制注入环境变量，优先于配置文件"""
    env_mappings = {
        "DEEPSEEK_KEY": ("api", "deepseek_key"),
        "XFYUN_KEY": ("api", "xfyun_key"),
        "SILICONFLOW_KEY": ("api", "siliconflow_key"),
    }
    for env_var, (section, key) in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value:
            config[section][key] = env_value
    return config
```

---

## 五、部署方案

### 5.1 Render 自动部署
```yaml
services:
  - type: web
    name: learning-assistant-plus
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
```

### 5.2 环境变量配置
| 变量名 | 说明 | 必需 |
|--------|------|------|
| DEEPSEEK_KEY | DeepSeek API Key | 是 |
| XFYUN_KEY | 讯飞星火 API Key | 是 |
| SILICONFLOW_KEY | SiliconFlow API Key | 是 |
| REPLICATE_KEY | Replicate API Key | 否 |
| ALIYUN_VIDEO_KEY | 通义万相 API Key | 否 |
| XFYUN_OCR_APPID | 讯飞 OCR AppID | 否 |
| XFYUN_OCR_APIKEY | 讯飞 OCR APIKey | 否 |
| XFYUN_OCR_SECRET | 讯飞 OCR Secret | 否 |

---

## 六、性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| API 路由数量 | 50+ | 覆盖所有功能 |
| 代码行数 | 2800+ | Python 后端 |
| 响应时间 | < 2s | 文本生成 |
| 图片生成 | < 10s | SiliconFlow |
| 视频生成 | < 60s | 通义万相 |
| OCR 识别 | < 3s | 讯飞 OCR |
| 数据库查询 | < 100ms | 索引优化后 |

---

## 七、创新点总结

### 7.1 技术创新
1. **多智能体协同架构**：14 个 Agent 协同工作，不是简单的 API 调用
2. **防幻觉机制**：6 维度检测 + 交叉验证，解决 AI 幻觉问题
3. **知识图谱构建**：自动提取知识点依赖，拓扑排序生成学习路径
4. **消息总线设计**：Agent 间通信可视化，便于调试和监控

### 7.2 工程创新
1. **API 重试机制**：指数退避重试，提高系统稳定性
2. **环境变量优先**：解决云平台重启后配置丢失问题
3. **数据库索引优化**：5 个索引提升查询性能
4. **输入验证过滤**：防 XSS 攻击，保障系统安全

### 7.3 应用创新
1. **完整学习闭环**：拍照搜题 → 智能答疑 → 资源生成 → 学习路径 → 学习评估
2. **个性化学习**：基于学生画像的动态学习方案
3. **多模态支持**：文本/图片/视频/OCR 全覆盖
4. **学习历史记录**：完整记录学习过程，便于回顾

---

## 八、项目文件结构

```
Learning-assistant-plus/
├── app.py                    # 核心后端（2800+ 行）
├── config.json               # 配置模板
├── requirements.txt          # 依赖清单
├── .gitignore               # Git 忽略规则
├── ARCHITECTURE.md          # 系统架构图
├── TECHNICAL_DOC.md         # 技术文档（本文件）
└── templates/
    ├── login.html           # 登录页面
    ├── admin.html           # 管理端页面
    └── user.html            # 用户端页面
```

---

## 九、团队分工

| 成员 | 职责 | 具体工作 |
|------|------|----------|
| 成员 A | 后端开发 | Flask 路由、API 集成、数据库设计、重试机制 |
| 成员 B | 前端开发 | HTML/CSS/JS、响应式设计、动画效果、错误处理 |
| 成员 C | 架构设计 | 多智能体架构、防幻觉机制、知识图谱、文档编写 |

---

## 十、未来展望

### 10.1 短期优化
- [ ] 添加单元测试
- [ ] 优化前端样式
- [ ] 增加更多学习主题
- [ ] 完善错误提示

### 10.2 长期规划
- [ ] 支持更多 AI 服务
- [ ] 引入向量数据库
- [ ] 实现实时协作学习
- [ ] 开发移动端 APP

---

**文档版本**：v1.0  
**最后更新**：2026-06-23  
**维护团队**：多智能体学习系统开发团队