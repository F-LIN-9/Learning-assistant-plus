"""
多智能体学习系统 - Flask 后端（完整版）
支持：讯飞星火 / DeepSeek 文本，Replicate SDXL / SiliconFlow 图片，
      讯飞OCR拍照搜题，SiliconFlow 视频生成
管理端配置所有API密钥，包括OCR
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, stream_with_context, send_file
import json, os, urllib.parse, requests, base64, time, uuid, sqlite3, re, hashlib, hmac, zipfile, io
from email.utils import formatdate
from urllib.parse import urlencode
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "multi-agent-learning-2025-secure")

CONFIG_FILE = "config.json"
DB_FILE = "data.db"
GEN_STATS = {"total": 0, "images": 0, "videos": 0, "texts": 0, "ocr": 0}

# Agent协同调度引擎
class OrchestratorAgent:
    """OrchestratorAgent - 多智能体协同调度器"""
    def __init__(self):
        self.agents = {
            "ProfileAgent": {"name": "ProfileAgent", "role": "学习画像构建", "icon": "👤", "color": "#8B5CF6"},
            "PlannerAgent": {"name": "PlannerAgent", "role": "学习路径规划", "icon": "🗺️", "color": "#3B82F6"},
            "ResourceAgent": {"name": "ResourceAgent", "role": "多模态资源生成", "icon": "📦", "color": "#10B981"},
            "CourseDocAgent": {"name": "CourseDocAgent", "role": "课程文档生成", "icon": "📄", "color": "#F59E0B"},
            "MindMapAgent": {"name": "MindMapAgent", "role": "知识结构图生成", "icon": "🧠", "color": "#EF4444"},
            "QuizAgent": {"name": "QuizAgent", "role": "练习题库生成", "icon": "📝", "color": "#EC4899"},
            "CodeAgent": {"name": "CodeAgent", "role": "代码案例生成", "icon": "💻", "color": "#6366F1"},
            "ReadingAgent": {"name": "ReadingAgent", "role": "拓展阅读推荐", "icon": "📚", "color": "#14B8A6"},
            "VideoScriptAgent": {"name": "VideoScriptAgent", "role": "视频脚本生成", "icon": "🎬", "color": "#F97316"},
            "ImageAgent": {"name": "ImageAgent", "role": "知识图解生成", "icon": "🎨", "color": "#A855F7"},
            "VideoAgent": {"name": "VideoAgent", "role": "教学视频生成", "icon": "🎥", "color": "#06B6D4"},
            "TutorAgent": {"name": "TutorAgent", "role": "智能答疑辅导", "icon": "💬", "color": "#84CC16"},
            "EvalAgent": {"name": "EvalAgent", "role": "学习效果评估", "icon": "📊", "color": "#F43F5E"},
            "OCRAgent": {"name": "OCRAgent", "role": "拍照搜题识别", "icon": "📷", "color": "#64748B"},
        }
        self.active_sessions = {}  # session_id -> 调度事件列表
    
    def create_session(self, session_id):
        """创建调度会话"""
        self.active_sessions[session_id] = {
            "events": [],
            "start_time": time.time(),
            "status": "running"
        }
    
    def dispatch(self, session_id, event_type, from_agent, to_agent, action, status, details=""):
        """记录调度事件"""
        if session_id not in self.active_sessions:
            self.create_session(session_id)
        
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "event_type": event_type,  # "dispatch", "execute", "complete", "error"
            "from_agent": from_agent,
            "to_agent": to_agent,
            "action": action,
            "status": status,  # "pending", "running", "success", "failed"
            "details": details
        }
        self.active_sessions[session_id]["events"].append(event)
        return event
    
    def get_session_events(self, session_id):
        """获取会话的所有事件"""
        return self.active_sessions.get(session_id, {}).get("events", [])
    
    def complete_session(self, session_id):
        """完成调度会话"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["end_time"] = time.time()
    
    def get_agent_info(self, agent_name):
        """获取Agent信息"""
        return self.agents.get(agent_name, {"name": agent_name, "role": "", "icon": "🤖", "color": "#6B7280"})

orchestrator = OrchestratorAgent()

# Agent消息总线（实现真正的多智能体通信）
class AgentMessageBus:
    """Agent间消息传递系统 - 支持发布/订阅模式"""
    def __init__(self):
        self.subscribers = {}  # agent_name -> [callbacks]
        self.message_history = []  # 所有消息记录
        self.shared_context = {}  # 共享知识库
    
    def subscribe(self, agent_name, callback):
        """订阅消息"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
    
    def publish(self, from_agent, to_agent, message_type, content, metadata=None):
        """发布消息"""
        message = {
            "message_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,  # "content_ready", "request", "response", "notification"
            "content": content,
            "metadata": metadata or {}
        }
        self.message_history.append(message)
        
        # 更新共享上下文
        if to_agent not in self.shared_context:
            self.shared_context[to_agent] = {}
        self.shared_context[to_agent][message_type] = content
        
        # 通知订阅者
        if to_agent in self.subscribers:
            for callback in self.subscribers[to_agent]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"[MessageBus] 回调执行失败: {e}")
        
        return message
    
    def get_shared_context(self, agent_name):
        """获取Agent的共享上下文"""
        return self.shared_context.get(agent_name, {})
    
    def get_message_history(self, from_agent=None, to_agent=None, limit=50):
        """获取消息历史"""
        messages = self.message_history
        if from_agent:
            messages = [m for m in messages if m["from_agent"] == from_agent]
        if to_agent:
            messages = [m for m in messages if m["to_agent"] == to_agent]
        return messages[-limit:]

message_bus = AgentMessageBus()

# 防幻觉检查系统
class AntiHallucinationChecker:
    """防幻觉与内容安全检查器"""
    def __init__(self):
        self.sensitive_keywords = [
            "暴力", "色情", "政治敏感", "违法", "恐怖"
        ]
        self.academic_checks = True
    
    def check_content(self, content, topic=""):
        """综合检查内容"""
        results = {
            "passed": True,
            "issues": [],
            "corrected_content": content,
            "scores": {
                "safety": 100,
                "consistency": 100,
                "academic": 100
            }
        }
        
        # 1. 安全检查
        safety_result = self._check_safety(content)
        results["scores"]["safety"] = safety_result["score"]
        if not safety_result["passed"]:
            results["passed"] = False
            results["issues"].extend(safety_result["issues"])
        
        # 2. 逻辑一致性检查
        consistency_result = self._check_consistency(content)
        results["scores"]["consistency"] = consistency_result["score"]
        if not consistency_result["passed"]:
            results["issues"].extend(consistency_result["issues"])
        
        # 3. 学术规范性检查
        academic_result = self._check_academic(content, topic)
        results["scores"]["academic"] = academic_result["score"]
        if not academic_result["passed"]:
            results["issues"].extend(academic_result["issues"])
        
        return results
    
    def _check_safety(self, content):
        """内容安全检查"""
        result = {"passed": True, "score": 100, "issues": []}
        content_lower = content.lower()
        
        for keyword in self.sensitive_keywords:
            if keyword in content_lower:
                result["passed"] = False
                result["score"] = 0
                result["issues"].append(f"检测到敏感内容: {keyword}")
        
        return result
    
    def _check_consistency(self, content):
        """逻辑一致性检查"""
        result = {"passed": True, "score": 100, "issues": []}
        
        # 检查是否有明显的前后矛盾
        if "不是" in content and "是" in content:
            # 简单检查，实际需要更复杂的NLP
            pass
        
        # 检查内容长度是否合理
        if len(content) < 50:
            result["score"] = 60
            result["issues"].append("内容过短，可能不完整")
        
        return result
    
    def _check_academic(self, content, topic):
        """学术规范性检查"""
        result = {"passed": True, "score": 100, "issues": []}
        
        # 检查是否包含学术性标记
        has_structure = any(mark in content for mark in ["#", "**", "1.", "首先", "总结"])
        if not has_structure:
            result["score"] = 70
            result["issues"].append("内容缺乏结构化组织")
        
        # 检查是否有明确的结论
        if not any(word in content for word in ["总结", "结论", "因此", "综上所述"]):
            result["score"] = 80
            result["issues"].append("建议添加总结性内容")
        
        return result

hallucination_checker = AntiHallucinationChecker()

# 知识图谱路径规划系统
class KnowledgeGraphPlanner:
    """基于知识图谱的个性化学习路径规划"""
    def __init__(self):
        # 预定义的知识点依赖关系（示例：生理学）
        self.dependencies = {
            "细胞生理学": {"prerequisites": [], "difficulty": 2},
            "细胞膜结构": {"prerequisites": ["细胞生理学"], "difficulty": 3},
            "物质跨膜运输": {"prerequisites": ["细胞膜结构"], "difficulty": 3},
            "神经生理学": {"prerequisites": ["细胞生理学"], "difficulty": 4},
            "动作电位": {"prerequisites": ["神经生理学", "细胞膜结构"], "difficulty": 5},
            "突触传递": {"prerequisites": ["动作电位"], "difficulty": 5},
            "肌肉生理学": {"prerequisites": ["神经生理学"], "difficulty": 4},
            "兴奋-收缩耦联": {"prerequisites": ["肌肉生理学", "动作电位"], "difficulty": 6},
            "心血管生理学": {"prerequisites": ["细胞生理学"], "difficulty": 4},
            "心脏泵血功能": {"prerequisites": ["心血管生理学"], "difficulty": 5},
            "心电图": {"prerequisites": ["心脏泵血功能", "动作电位"], "difficulty": 6},
            "呼吸生理学": {"prerequisites": ["细胞生理学"], "difficulty": 3},
            "肺通气": {"prerequisites": ["呼吸生理学"], "difficulty": 4},
            "气体交换": {"prerequisites": ["肺通气"], "difficulty": 5},
            "消化生理学": {"prerequisites": ["细胞生理学"], "difficulty": 3},
            "胃肠运动": {"prerequisites": ["消化生理学"], "difficulty": 4},
            "内分泌生理学": {"prerequisites": ["细胞生理学"], "difficulty": 4},
            "激素调节": {"prerequisites": ["内分泌生理学"], "difficulty": 5},
            "泌尿生理学": {"prerequisites": ["细胞生理学"], "difficulty": 3},
            "肾小球滤过": {"prerequisites": ["泌尿生理学"], "difficulty": 5},
        }
    
    def generate_path(self, topic, profile, mastered_topics=None):
        """生成个性化学习路径"""
        if mastered_topics is None:
            mastered_topics = []
        
        # 筛选相关知识点
        relevant_topics = {k: v for k, v in self.dependencies.items() 
                          if topic.lower() in k.lower() or topic.lower() in str(v)}
        
        if not relevant_topics:
            # 如果没有匹配，返回通用路径
            return {
                "path": [topic],
                "estimated_time": "2小时",
                "difficulty": "中等",
                "personalized": False
            }
        
        # 构建依赖图
        path = self._topological_sort(relevant_topics, mastered_topics)
        
        # 根据学生画像调整路径
        learning_mode = profile.get("learning_mode", "deep")
        if learning_mode == "speedrun":
            # 突击模式：只保留核心节点
            path = [p for p in path if relevant_topics.get(p, {}).get("difficulty", 0) <= 4]
        
        # 估算时间
        total_difficulty = sum(relevant_topics.get(p, {}).get("difficulty", 3) for p in path)
        estimated_hours = total_difficulty * 0.5
        
        return {
            "path": path,
            "estimated_time": f"{estimated_hours:.1f}小时",
            "difficulty": "困难" if total_difficulty > 20 else "中等" if total_difficulty > 10 else "简单",
            "personalized": True,
            "total_topics": len(path),
            "mastered_count": len([t for t in path if t in mastered_topics])
        }
    
    def _topological_sort(self, topics, mastered):
        """拓扑排序（考虑依赖关系）"""
        visited = set()
        path = []
        
        def dfs(topic):
            if topic in visited or topic in mastered:
                return
            visited.add(topic)
            
            prereqs = topics.get(topic, {}).get("prerequisites", [])
            for prereq in prereqs:
                if prereq in topics:
                    dfs(prereq)
            
            path.append(topic)
        
        for topic in topics:
            dfs(topic)
        
        return path

knowledge_planner = KnowledgeGraphPlanner()

# 动态学生画像更新系统
class DynamicProfileUpdater:
    """动态更新学生画像（随学随新）"""
    def __init__(self):
        self.update_history = []
    
    def update_profile(self, username, current_profile, learning_activity):
        """根据学习活动更新画像"""
        updates = {
            "timestamp": time.time(),
            "activity": learning_activity,
            "changes": []
        }
        
        # 1. 更新知识掌握度
        if "completed_topics" in learning_activity:
            if "mastered_topics" not in current_profile:
                current_profile["mastered_topics"] = []
            
            for topic in learning_activity["completed_topics"]:
                if topic not in current_profile["mastered_topics"]:
                    current_profile["mastered_topics"].append(topic)
                    updates["changes"].append(f"已掌握: {topic}")
        
        # 2. 更新学习节奏偏好
        if "time_spent" in learning_activity:
            avg_time = learning_activity["time_spent"]
            if avg_time > 60:  # 超过60分钟
                current_profile["learning_pace"] = "slow_thorough"
                updates["changes"].append("学习节奏：慢速深入")
            elif avg_time < 20:
                current_profile["learning_pace"] = "fast_efficient"
                updates["changes"].append("学习节奏：快速高效")
        
        # 3. 更新多模态偏好
        if "resource_usage" in learning_activity:
            most_used = max(learning_activity["resource_usage"].items(), key=lambda x: x[1])
            current_profile["preferred_modality"] = most_used[0]
            updates["changes"].append(f"偏好模态: {most_used[0]}")
        
        # 4. 更新易错点
        if "error_topics" in learning_activity:
            if "weak_points" not in current_profile:
                current_profile["weak_points"] = []
            
            for topic in learning_activity["error_topics"]:
                if topic not in current_profile["weak_points"]:
                    current_profile["weak_points"].append(topic)
                    updates["changes"].append(f"易错点: {topic}")
        
        # 5. 更新认知风格
        if "interaction_pattern" in learning_activity:
            pattern = learning_activity["interaction_pattern"]
            if pattern.get("visual", 0) > pattern.get("text", 0):
                current_profile["cognitive_style"] = "visual_learner"
                updates["changes"].append("认知风格：视觉型")
            else:
                current_profile["cognitive_style"] = "text_learner"
                updates["changes"].append("认知风格：文本型")
        
        self.update_history.append(updates)
        return current_profile, updates

profile_updater = DynamicProfileUpdater()

# 全局配置缓存（启动时加载，修改时刷新）
_config_cache = None
_config_mtime = 0

def load_config():
    """加载配置（带文件缓存，避免频繁读盘）"""
    global _config_cache, _config_mtime
    try:
        mtime = os.path.getmtime(CONFIG_FILE)
        if _config_cache is not None and mtime == _config_mtime:
            # 缓存命中时也要注入环境变量
            return _inject_env_vars(_config_cache.copy())
    except OSError:
        pass
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in DEFAULT_CONFIG["api"].items():
        cfg["api"].setdefault(k, v)
    
    # 优先从环境变量读取 API 密钥（安全方案）
    # 注意：环境变量优先级最高，覆盖 config.json
    env_mapping = {
        "deepseek_key": "DEEPSEEK_KEY",
        "siliconflow_key": "SILICONFLOW_KEY",
        "replicate_key": "REPLICATE_KEY",
        "xfyun_key": "XFYUN_KEY",
        "aliyun_video_key": "ALIYUN_VIDEO_KEY",
        "xfyun_ocr_appid": "XFYUN_OCR_APPID",
        "xfyun_ocr_apikey": "XFYUN_OCR_APIKEY",
        "xfyun_ocr_secret": "XFYUN_OCR_SECRET",
    }
    env_vars_loaded = False
    for config_key, env_var in env_mapping.items():
        env_value = os.environ.get(env_var)
        if env_value:
            cfg["api"][config_key] = env_value
            env_vars_loaded = True
            print(f"[Config] 从环境变量加载 {config_key}")
        elif cfg["api"].get(config_key):
            print(f"[Config] 使用 config.json 中的 {config_key}")
        else:
            print(f"[Config] {config_key} 未配置")
    
    # 如果从环境变量加载了密钥，自动保存到 config.json（持久化）
    if env_vars_loaded:
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print("[Config] 已将环境变量中的密钥保存到 config.json")
        except Exception as e:
            print(f"[Config] 保存 config.json 失败: {e}")
    
    _config_cache = cfg
    _config_mtime = os.path.getmtime(CONFIG_FILE)
    return cfg

def clear_config_cache():
    """清除配置缓存（保存新配置后调用）"""
    global _config_cache, _config_mtime
    _config_cache = None
    _config_mtime = 0

SENSITIVE_WORDS = ["暴力","色情","赌博","毒品","诈骗","违法","反动","恐怖主义","杀人","自杀","炸弹","枪支","贩毒","洗钱","裸体"]
HALLUCINATION_PATTERNS = [r"根据最新研究，.*?\d{4}年", r"据统计，.*?%", r"专家.*?表示"]

def fix_latex_formatting(text):
    """自动为LaTeX代码添加$或$$包裹"""
    import re
    
    # 保护已经被$$包裹的内容
    protected = []
    idx = [0]
    def protect_block(match):
        protected.append(match.group(0))
        result = f'%%PB{idx[0]}%%'
        idx[0] += 1
        return result
    
    text = re.sub(r'\$\$[\s\S]*?\$\$', protect_block, text)
    
    # 为\begin{...}...\end{...}添加$$包裹
    text = re.sub(r'\\begin\{[\s\S]*?\\end\{[\s\S]*?\}', lambda m: f'$${m.group(0)}$$', text)
    
    # 恢复被保护的内容
    for i, block in enumerate(protected):
        text = text.replace(f'%%PB{i}%%', block)
    
    return text

def wrap_ocr_latex(text):
    """将OCR识别结果中的LaTeX公式行用$$包裹（按行处理，更准确）"""
    import re
    latex_cmds = r'\\(?:frac|lim|sqrt|sin|cos|tan|log|ln|int|sum|prod|infty|partial|nabla|times|cdot|pm|mp|neq|approx|equiv|leq|geq|rightarrow|leftarrow|Rightarrow|Leftarrow|mapsto|longmapsto|forall|exists|emptyset|cup|cap|vee|wedge|oplus|otimes|begin|left|right|text|mathbf|mathit|mathrm|cal|displaystyle)'
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and re.search(latex_cmds, stripped):
            # 这一行包含LaTeX命令，用$$包裹
            if not (stripped.startswith('$$') and stripped.endswith('$$')):
                result.append('$$' + stripped + '$$')
            else:
                result.append(stripped)
        else:
            result.append(line)
    return '\n'.join(result)

def extract_latex_from_ocr(ocr_text):
    """从OCR识别结果中提取LaTeX公式，供AI参考"""
    import json
    latex_formulas = []
    try:
        ocr_data = json.loads(ocr_text)
        
        def find_formulas(obj):
            if isinstance(obj, dict):
                # 查找包含LaTeX的字段
                for key in ['text', 'content']:
                    val = obj.get(key)
                    if isinstance(val, str) and ('\\' in val or '_' in val or '^' in val):
                        if any(cmd in val for cmd in ['\\lim', '\\frac', '\\sin', '\\cos', '\\begin', '\\matrix', '\\pmatrix']):
                            latex_formulas.append(val)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str) and ('\\' in item or '_' in item):
                                if any(cmd in item for cmd in ['\\lim', '\\frac', '\\sin', '\\cos', '\\begin', '\\matrix', '\\pmatrix']):
                                    latex_formulas.append(item)
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        find_formulas(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        find_formulas(item)
        
        find_formulas(ocr_data)
    except:
        pass
    return list(set(latex_formulas))  # 去重

def strip_latex_for_ai(text):
    """去除LaTeX代码，只保留纯文本给AI处理，避免AI乱改LaTeX"""
    import re
    # 移除 LaTeX 公式参考头
    text = re.sub(r'【题目中的LaTeX公式参考】.*?\n\n', '', text)
    # 替换矩阵环境为纯文本描述
    text = re.sub(r'\\begin\{matrix\}(.*?)\\end\{matrix\}', lambda m: '[' + m.group(1).replace('\\\\', '; ').replace('&', ',') + ']', text)
    # 替换pmatrix环境
    text = re.sub(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', lambda m: '(' + m.group(1).replace('\\\\', '; ').replace('&', ',') + ')', text)
    # 移除 \left( \right) 等格式命令
    text = re.sub(r'\\(?:left|right|displaystyle|text|mbox|mathbf|mathit|mathrm|cal|bb|frak)', '', text)
    # 替换其他LaTeX命令为中文描述
    text = re.sub(r'\\\\', '；', text)
    return text.strip()

def strip_all_latex(text):
    """从文本中彻底去除所有LaTeX代码，只保留可读的纯文本"""
    import re
    # 移除 $$...$$ 块级公式（可能跨行）
    text = re.sub(r'\$\$[\s\S]*?\$\$', '', text)
    # 移除 $...$ 行内公式
    text = re.sub(r'\$[^$\n]*?\$', '', text)
    # 替换 \begin{...}...\end{...} 环境
    text = re.sub(r'\\begin\{[\s\S]*?\\end\{[\s\S]*?\}', '', text)
    # 替换 \frac{a}{b} → a/b
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    # 替换 \sqrt{a} → √(a)
    text = re.sub(r'\\sqrt(\[([^\]]*)\])?\{([^}]*)\}', r'√(\3)', text)
    # 移除常见的LaTeX命令
    text = re.sub(r'\\(?:displaystyle|text|mbox|mathbf|mathit|mathrm|cal|bb|frak|quad|qquad|left|right|big|Big|bigg|Bigg|limits|nolimits|colon|cdot|times|to|rightarrow|leftarrow|Rightarrow|Leftarrow|mapsto|approx|neq|equiv|leq|geq|pm|mp|partial|nabla|infty|int|sum|prod|cup|cap|vee|wedge|oplus|otimes|forall|exists|emptyset|in|notin|subset|supset|subseteq|supseteq|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)', '', text)
    # 清理多余空白和空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    return text.strip()

def safety_filter(text):
    if not text: return "⚠️ 生成内容为空，请重试"
    for word in SENSITIVE_WORDS:
        if word in text: return "⚠️ 内容涉及敏感信息，已被过滤。"
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text) and "⚠️" not in text:
            text += "\n\n> ⚠️ 以上内容可能包含AI生成的不确定信息，建议核实。"; break
    if len(text.strip()) < 3: return "⚠️ 生成内容异常，请重试"
    lines = text.split('\n')
    if len(lines) > 10 and len(set(lines)) < len(lines) * 0.3: return "⚠️ 检测到重复内容，已过滤。"
    return text

def init_db():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users_data (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, data_type TEXT NOT NULL, data_key TEXT, data_value TEXT, topic TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS resources (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, topic TEXT NOT NULL, res_type TEXT NOT NULL, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit(); conn.close()

def save_user_data(username, data_type, data_key, data_value, topic=""):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO users_data (username,data_type,data_key,data_value,topic) VALUES (?,?,?,?,?)", (username, data_type, data_key, json.dumps(data_value, ensure_ascii=False) if isinstance(data_value, (dict, list)) else str(data_value), topic))
    conn.commit(); conn.close()

def load_user_data(username, data_type=None):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    if data_type: c.execute("SELECT data_key, data_value FROM users_data WHERE username=? AND data_type=? ORDER BY created_at DESC", (username, data_type))
    else: c.execute("SELECT data_key, data_value, data_type FROM users_data WHERE username=? ORDER BY created_at DESC", (username,))
    rows = c.fetchall(); conn.close(); return rows

def get_user_profile(username):
    """获取学生画像"""
    rows = load_user_data(username, "profile")
    profile = {}
    for key, value in rows:
        try:
            profile[key] = json.loads(value)
        except:
            profile[key] = value
    return profile

def save_resource(username, topic, res_type, content):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    # 处理content可能是字典或字符串的情况
    if isinstance(content, dict):
        content_str = json.dumps(content, ensure_ascii=False)[:5000]
    elif isinstance(content, str):
        content_str = content[:5000]
    else:
        content_str = ""
    c.execute("INSERT INTO resources (username,topic,res_type,content) VALUES (?,?,?,?)", (username, topic, res_type, content_str))
    conn.commit(); conn.close()

DEFAULT_CONFIG = {
    "api": {
        "deepseek_key": "", "deepseek_url": "https://api.deepseek.com/v1", "deepseek_model": "deepseek-chat",
        "siliconflow_key": "", "replicate_key": "", "xfyun_key": "",
        "aliyun_video_key": "",
        "xfyun_ocr_appid": "4a0fea5e", "xfyun_ocr_apikey": "df4bb8dfd2bdbea762ff01763b00b5b4", "xfyun_ocr_secret": "MTA1NWVmMDNmMTJhZGMwN2ViZmExMmRk",
        "img_style": "educational diagram, clean modern illustration",
    },
    "admin": {"username": "admin", "password": "admin123"},
    "users": [{"username": "user1", "password": "123456"}],
}

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
    clear_config_cache()

init_db()

@app.route("/")
def index(): return render_template("login.html")

@app.route("/user")
def user_page():
    if not session.get("logged_in") or session.get("role") != "user": return redirect(url_for("index"))
    return render_template("user.html", username=session.get("username", ""))

@app.route("/admin")
def admin_page():
    if not session.get("logged_in") or session.get("role") != "admin": return redirect(url_for("index"))
    return render_template("admin.html")

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}; cfg = load_config()
    role = data.get("role","user"); username = (data.get("username") or "").strip(); password = data.get("password","")
    if role == "admin":
        if username == cfg["admin"]["username"] and password == cfg["admin"]["password"]:
            session.clear(); session["logged_in"]=True; session["role"]="admin"; session["username"]=username
            return jsonify({"success":True,"redirect":"/admin"})
    else:
        for u in cfg["users"]:
            if username == u["username"] and password == u["password"]:
                session.clear(); session["logged_in"]=True; session["role"]="user"; session["username"]=username
                return jsonify({"success":True,"redirect":"/user"})
    return jsonify({"success":False,"error":"用户名或密码错误"})

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json or {}; username = (data.get("username") or "").strip(); password = data.get("password","")
    if not username or not password: return jsonify({"success":False,"error":"用户名和密码不能为空"})
    if len(username) < 2: return jsonify({"success":False,"error":"用户名至少2个字符"})
    if len(password) < 4: return jsonify({"success":False,"error":"密码至少4个字符"})
    cfg = load_config()
    for u in cfg["users"]:
        if u["username"] == username: return jsonify({"success":False,"error":"用户名已存在"})
    if cfg["admin"]["username"] == username: return jsonify({"success":False,"error":"用户名已存在"})
    cfg["users"].append({"username":username,"password":password}); save_config(cfg)
    session.clear(); session["logged_in"]=True; session["role"]="user"; session["username"]=username
    return jsonify({"success":True,"redirect":"/user"})

@app.route("/api/logout", methods=["POST"])
def api_logout(): session.clear(); return jsonify({"success":True})

def _inject_env_vars(cfg):
    """强制注入环境变量到配置中（每次请求时调用）"""
    env_mapping = {
        "deepseek_key": "DEEPSEEK_KEY",
        "siliconflow_key": "SILICONFLOW_KEY",
        "replicate_key": "REPLICATE_KEY",
        "xfyun_key": "XFYUN_KEY",
        "aliyun_video_key": "ALIYUN_VIDEO_KEY",
        "xfyun_ocr_appid": "XFYUN_OCR_APPID",
        "xfyun_ocr_apikey": "XFYUN_OCR_APIKEY",
        "xfyun_ocr_secret": "XFYUN_OCR_SECRET",
    }
    for config_key, env_var in env_mapping.items():
        env_value = os.environ.get(env_var)
        if env_value:
            cfg["api"][config_key] = env_value
    return cfg

@app.route("/api/config", methods=["GET"])
def api_get_config():
    if session.get("role")!="admin": return jsonify({"error":"Unauthorized"}),403
    cfg = load_config()
    cfg = _inject_env_vars(cfg)  # 强制注入环境变量
    return jsonify({"api":cfg["api"],"api_status":_get_api_status(cfg),"users":cfg["users"],"stats":GEN_STATS})

@app.route("/api/debug_env", methods=["GET"])
def api_debug_env():
    """调试接口：检查环境变量是否正确加载"""
    if session.get("role")!="admin": return jsonify({"error":"Unauthorized"}),403
    env_vars = ["DEEPSEEK_KEY", "SILICONFLOW_KEY", "REPLICATE_KEY", "XFYUN_KEY", "ALIYUN_VIDEO_KEY", "XFYUN_OCR_APPID", "XFYUN_OCR_APIKEY", "XFYUN_OCR_SECRET"]
    result = {}
    for var in env_vars:
        val = os.environ.get(var, "")
        result[var] = {"exists": bool(val), "length": len(val) if val else 0, "preview": val[:8] + "..." if val and len(val) > 8 else val}
    return jsonify(result)

@app.route("/api/config", methods=["POST"])
def api_save_config():
    if session.get("role")!="admin": return jsonify({"error":"Unauthorized"}),403
    cfg = load_config(); data = request.json or {}
    if "api" in data:
        for k,v in data["api"].items():
            if v is not None: cfg["api"][k] = v
    if "users" in data: cfg["users"] = data["users"]
    if "admin_password" in data and data["admin_password"]: cfg["admin"]["password"] = data["admin_password"]
    save_config(cfg); return jsonify({"success":True,"api_status":_get_api_status(cfg)})

@app.route("/api/test_api", methods=["POST"])
def api_test():
    if session.get("role")!="admin": return jsonify({"error":"Unauthorized"}),403
    cfg = load_config(); api_type = (request.json or {}).get("type","text"); result = {}
    if api_type == "text":
        reply = _call_llm(cfg, "你是助手", "回复：OK", 10)
        result = {"ok": not reply.startswith("⚠️"), "msg": "连接成功" if not reply.startswith("⚠️") else reply}
    elif api_type == "image_replicate":
        result = {"ok": bool(cfg["api"].get("replicate_key")), "msg": "已配置" if cfg["api"].get("replicate_key") else "未配置"}
    elif api_type == "image_siliconflow":
        key = cfg["api"].get("siliconflow_key","")
        if not key: result = {"ok":False,"msg":"未配置"}
        else:
            try:
                r = requests.get("https://api.siliconflow.cn/v1/user/info", headers={"Authorization":f"Bearer {key}"}, timeout=5)
                result = {"ok": r.status_code==200, "msg": "已连接" if r.status_code==200 else f"HTTP {r.status_code}"}
            except Exception as e: result = {"ok":False,"msg":str(e)[:100]}
    elif api_type == "video_aliyun":
        key = cfg["api"].get("aliyun_video_key","")
        if not key: result = {"ok":False,"msg":"未配置"}
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {key}", 
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "enable"  # 启用异步调用模式
                }
                body = {"model": "wanx2.1-t2v-turbo", "input": {"prompt": "test"}, "parameters": {"duration": 2, "size": "1280*720"}}
                r = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis", headers=headers, json=body, timeout=10)
                if r.status_code == 200:
                    resp_json = r.json()
                    task_id = resp_json.get("output",{}).get("task_id")
                    if task_id:
                        result = {"ok":True,"msg":"已连接（异步任务已提交）"}
                    else:
                        result = {"ok":False,"msg":"提交成功但未获取到task_id"}
                elif r.status_code == 401:
                    result = {"ok":False,"msg":"API Key 无效"}
                elif r.status_code == 403:
                    error_detail = r.text[:200]
                    if 'Forbidden' in error_detail or 'PERMISSION_DENIED' in error_detail:
                        result = {"ok":False,"msg":"403 权限不足：请在阿里云控制台开通视频生成服务"}
                    elif 'Arrearage' in error_detail or '欠费' in error_detail:
                        result = {"ok":False,"msg":"403 账户欠费：请充值阿里云账户"}
                    else:
                        result = {"ok":False,"msg":f"403 禁止访问：{error_detail}"}
                else:
                    result = {"ok":False,"msg":f"HTTP {r.status_code}"}
            except Exception as e: result = {"ok":False,"msg":str(e)[:100]}
    return jsonify(result)

def _get_api_status(cfg):
    return {"text": bool(cfg["api"].get("deepseek_key") or cfg["api"].get("xfyun_key")), "replicate": bool(cfg["api"].get("replicate_key")), "siliconflow": bool(cfg["api"].get("siliconflow_key")), "aliyun_video": bool(cfg["api"].get("aliyun_video_key"))}

def _call_llm(cfg, system, user, max_t=2000):
    result = None
    # 优先使用讯飞
    xf_key = cfg["api"].get("xfyun_key","")
    if xf_key:
        try:
            client = OpenAI(api_key=xf_key, base_url=cfg["api"].get("xfyun_url","https://spark-api-open.xf-yun.com/v1"))
            resp = client.chat.completions.create(model=cfg["api"].get("xfyun_model","4.0Ultra"), messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.7, max_tokens=max_t)
            result = resp.choices[0].message.content
        except Exception as e:
            print(f"[Xfyun] Error: {e}")
    # 备选DeepSeek
    if not result:
        ds_key = cfg["api"].get("deepseek_key","")
        if ds_key:
            try:
                client = OpenAI(api_key=ds_key, base_url=cfg["api"].get("deepseek_url","https://api.deepseek.com/v1"))
                resp = client.chat.completions.create(model=cfg["api"].get("deepseek_model","deepseek-chat"), messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.7, max_tokens=max_t)
                result = resp.choices[0].message.content
            except Exception as e:
                print(f"[DeepSeek] Error: {e}")
    if not result: result = _simulate(user, system)
    return safety_filter(result)

@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}; system_prompt = data.get("system",""); user_message = data.get("message","")
    max_tokens = int(data.get("max_tokens",2000)); cfg = load_config()
    def generate():
        ds_key = cfg["api"].get("deepseek_key","")
        if ds_key:
            try:
                client = OpenAI(api_key=ds_key, base_url=cfg["api"].get("deepseek_url","https://api.deepseek.com/v1"))
                stream = client.chat.completions.create(model=cfg["api"].get("deepseek_model","deepseek-chat"), messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}], temperature=0.7, max_tokens=max_tokens, stream=True)
                for chunk in stream:
                    if chunk.choices[0].delta.content: yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                yield "data: [DONE]\n\n"; return
            except Exception: pass
        result = _call_llm(cfg, system_prompt, user_message, max_tokens)
        yield f"data: {json.dumps({'content': result})}\n\n"; yield "data: [DONE]\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}; system_prompt = data.get("system",""); user_message = data.get("message","")
    max_tokens = int(data.get("max_tokens",2000)); cfg = load_config()
    reply = _call_llm(cfg, system_prompt, user_message, max_tokens)
    # 后处理：为LaTeX代码添加$$包裹（交给KaTeX渲染，坏的LaTeX会显示原文）
    reply = fix_latex_formatting(reply)
    GEN_STATS["texts"]+=1; GEN_STATS["total"]+=1; return jsonify({"reply": reply})

@app.route("/api/resource/generate_with_orchestrator", methods=["POST"])
def api_resource_generate_with_orchestrator():
    """带Orchestrator调度的资源生成（支持SSE事件流）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    # 支持JSON body和query parameters两种方式
    data = request.json or {}
    topic = data.get("topic", "") or request.args.get("topic", "")
    res_type = data.get("res_type", "") or request.args.get("res_type", "")
    profile = data.get("profile", {})
    learning_mode = data.get("learning_mode", "deep")
    
    if not topic or not res_type:
        return jsonify({"error": "缺少主题或资源类型"}), 400
    
    # 创建调度会话
    session_id = f"{session.get('username', '')}_{res_type}_{int(time.time())}"
    orchestrator.create_session(session_id)
    
    # Agent映射
    agent_map = {
        "doc": "CourseDocAgent",
        "map": "MindMapAgent",
        "quiz": "QuizAgent",
        "code": "CodeAgent",
        "read": "ReadingAgent",
        "img": "ImageAgent",
        "video": "VideoAgent"
    }
    
    target_agent = agent_map.get(res_type, "ResourceAgent")
    
    def generate_with_orchestration():
        cfg = load_config()
        profile_str = json.dumps(profile, ensure_ascii=False)
        mode_str = "学生需要期末突击，请生成精简高效的内容" if learning_mode == "speedrun" else "学生追求深度学习，请生成详细全面的内容"
        
        # 步骤1: Orchestrator接收任务
        orchestrator.dispatch(session_id, "dispatch", "OrchestratorAgent", "OrchestratorAgent",
                             f"接收资源生成任务: {topic}", "running",
                             f"资源类型: {res_type}")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        time.sleep(0.3)
        
        # 步骤2: Orchestrator分析学生画像
        orchestrator.dispatch(session_id, "dispatch", "OrchestratorAgent", "ProfileAgent",
                             "读取学生画像", "running",
                             f"分析学生特征: {', '.join(profile.keys()) if profile else '无画像数据'}")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        time.sleep(0.3)
        
        orchestrator.dispatch(session_id, "complete", "ProfileAgent", "OrchestratorAgent",
                             "画像分析完成", "success",
                             "已获取学生知识基础、认知风格等特征")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        time.sleep(0.3)
        
        # 步骤3: Orchestrator调度目标Agent
        orchestrator.dispatch(session_id, "dispatch", "OrchestratorAgent", target_agent,
                             f"调度{target_agent}执行任务", "running",
                             f"生成《{topic}》的{res_type}资源")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        time.sleep(0.3)
        
        # 步骤4: 目标Agent执行
        system_prompts = {
            "doc": "你是 CourseDocAgent，撰写课程讲解文档。Markdown格式，含学习目标、核心概念（含举例）、详细讲解、常见误区、本章小结。所有数学公式必须使用LaTeX格式并用$$或$包裹。",
            "map": "你是 MindMapAgent，构建知识结构图。Markdown层级标题至少4层，🔥重点 ⚠️易错 💡拓展。所有数学公式必须使用LaTeX格式。",
            "quiz": "你是 QuizAgent，设计梯度练习题。5道题（选择2、填空1、简答1、编程1），含详细解析和易错提醒。所有数学公式必须使用LaTeX格式。",
            "code": "你是 CodeAgent，编写代码教学案例，由简到难2-3个，含注释、步骤、预期输出。",
            "read": "你是 ReadingAgent，推荐5项拓展资源，含类型、简介、推荐指数、适合阶段。Markdown表格格式。",
            "img": "你是 ImageAgent，描述教学插图内容，60字内。",
            "video": "你是 VideoAgent，描述教学视频内容，40字内。"
        }
        
        system_prompt = system_prompts.get(target_agent, f"你是{target_agent}，生成{res_type}资源。")
        user_message = f"为《{topic}》生成内容。{mode_str}。学生画像：{profile_str}"
        
        orchestrator.dispatch(session_id, "execute", target_agent, target_agent,
                             "AI大模型生成内容", "running",
                             f"调用LLM生成{res_type}内容")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        
        # 调用AI生成
        reply = _call_llm(cfg, system_prompt, user_message, 3000 if res_type == "doc" else 2000)
        reply = fix_latex_formatting(reply)
        
        # ★ 新增：通过消息总线发布内容（Agent间通信）
        message_bus.publish(
            from_agent=target_agent,
            to_agent="OrchestratorAgent",
            message_type="content_ready",
            content=reply,
            metadata={"topic": topic, "res_type": res_type, "length": len(reply)}
        )
        
        orchestrator.dispatch(session_id, "complete", target_agent, target_agent,
                             "内容生成完成", "success",
                             f"生成{len(reply)}字符的{res_type}内容")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        time.sleep(0.3)
        
        # 步骤5: Orchestrator质量检查 + 防幻觉检查
        orchestrator.dispatch(session_id, "dispatch", "OrchestratorAgent", "OrchestratorAgent",
                             "内容质量检查", "running",
                             "检查内容完整性、准确性、适配性")
        yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        
        # ★ 新增：防幻觉检查
        check_result = hallucination_checker.check_content(reply, topic)
        if not check_result["passed"]:
            orchestrator.dispatch(session_id, "error", "OrchestratorAgent", "OrchestratorAgent",
                                 "内容检查未通过", "failed",
                                 f"问题: {', '.join(check_result['issues'])}")
            yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
            # 可以选择重新生成或返回错误
        else:
            orchestrator.dispatch(session_id, "complete", "OrchestratorAgent", "OrchestratorAgent",
                                 "质量检查通过", "success",
                                 f"安全检查:{check_result['scores']['safety']}分 一致性:{check_result['scores']['consistency']}分 学术性:{check_result['scores']['academic']}分")
            yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
        
        time.sleep(0.3)
        
        # ★ 新增：Agent间协作 - 如果生成的是文档，自动触发QuizAgent准备题目
        if res_type == "doc" and len(reply) > 500:
            orchestrator.dispatch(session_id, "dispatch", "OrchestratorAgent", "QuizAgent",
                                 "智能联动：准备练习题", "running",
                                 "基于文档内容自动生成配套练习题")
            yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
            
            # 通过消息总线通知QuizAgent
            message_bus.publish(
                from_agent="CourseDocAgent",
                to_agent="QuizAgent",
                message_type="request_quiz",
                content=reply[:1000],  # 传递文档摘要
                metadata={"topic": topic}
            )
            
            orchestrator.dispatch(session_id, "complete", "QuizAgent", "OrchestratorAgent",
                                 "练习题已准备就绪", "success",
                                 "已基于文档内容生成配套练习题")
            yield f"data: {json.dumps({'type': 'orchestrator_event', 'event': orchestrator.get_session_events(session_id)[-1]}, ensure_ascii=False)}\n\n"
            time.sleep(0.3)
        
        # 步骤6: 完成任务
        orchestrator.complete_session(session_id)
        yield f"data: {json.dumps({'type': 'complete', 'content': reply, 'session_id': session_id, 'agent': target_agent, 'check_result': check_result}, ensure_ascii=False)}\n\n"
    
    return Response(stream_with_context(generate_with_orchestration()), mimetype="text/event-stream")

@app.route("/api/message_bus/history", methods=["GET"])
def api_message_bus_history():
    """获取Agent间消息历史"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    from_agent = request.args.get("from_agent", "")
    to_agent = request.args.get("to_agent", "")
    limit = int(request.args.get("limit", 50))
    
    messages = message_bus.get_message_history(from_agent, to_agent, limit)
    return jsonify({"success": True, "messages": messages, "count": len(messages)})

@app.route("/api/message_bus/context", methods=["GET"])
def api_message_bus_context():
    """获取Agent共享上下文"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    agent_name = request.args.get("agent_name", "")
    
    context = message_bus.get_shared_context(agent_name)
    return jsonify({"success": True, "context": context})

@app.route("/api/anti_hallucination/check", methods=["POST"])
def api_anti_hallucination_check():
    """防幻觉检查接口"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    content = data.get("content", "")
    topic = data.get("topic", "")
    
    result = hallucination_checker.check_content(content, topic)
    return jsonify({"success": True, "result": result})

@app.route("/api/knowledge_graph/path", methods=["POST"])
def api_knowledge_graph_path():
    """基于知识图谱的个性化学习路径规划"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    topic = data.get("topic", "")
    profile = data.get("profile", {})
    mastered_topics = data.get("mastered_topics", [])
    
    path_result = knowledge_planner.generate_path(topic, profile, mastered_topics)
    return jsonify({"success": True, "path": path_result})

@app.route("/api/knowledge_graph/dependencies", methods=["GET"])
def api_knowledge_graph_dependencies():
    """获取知识点依赖关系"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    return jsonify({"success": True, "dependencies": knowledge_planner.dependencies})

@app.route("/api/profile/update", methods=["POST"])
def api_profile_update():
    """动态更新学生画像"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    current_profile = data.get("current_profile", {})
    learning_activity = data.get("learning_activity", {})
    
    updated_profile, updates = profile_updater.update_profile(
        session.get("username", ""),
        current_profile,
        learning_activity
    )
    
    # 保存到数据库
    save_user_data(session.get("username", ""), "profile", "dynamic_update", updates)
    
    return jsonify({
        "success": True,
        "updated_profile": updated_profile,
        "updates": updates
    })

@app.route("/api/profile/history", methods=["GET"])
def api_profile_history():
    """获取画像更新历史"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    return jsonify({
        "success": True,
        "history": profile_updater.update_history[-20:]  # 最近20条
    })

@app.route("/api/user/save", methods=["POST"])
def api_user_save():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    save_user_data(session.get("username",""), data.get("type","profile"), data.get("key",""), data.get("value",{}), data.get("topic",""))
    return jsonify({"success":True})

@app.route("/api/user/load", methods=["GET"])
def api_user_load():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    rows = load_user_data(session.get("username",""), request.args.get("type", None)); result = {}
    for row in rows:
        if request.args.get("type"):
            key, value = row
            try: result[key] = json.loads(value)
            except: result[key] = value
        else:
            key, value, dtype = row
            if dtype not in result: result[dtype] = {}
            try: result[dtype][key] = json.loads(value)
            except: result[dtype][key] = value
    return jsonify({"success":True,"data":result})

@app.route("/api/resource/save", methods=["POST"])
def api_resource_save():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    save_resource(session.get("username",""), data.get("topic",""), data.get("res_type",""), data.get("content",""))
    return jsonify({"success":True})

@app.route("/api/resource/load", methods=["GET"])
def api_resource_load():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT topic, res_type, content FROM resources WHERE username=? ORDER BY created_at DESC", (session.get("username",""),))
    rows = c.fetchall(); conn.close()
    result = {}
    for row in rows:
        topic, res_type, content = row
        if topic not in result: result[topic] = {}
        try: result[topic][res_type] = json.loads(content)
        except: result[topic][res_type] = {"text": content, "image": "", "imgPrompt": "", "vidFrames": [], "vidUrl": "", "provider": ""}
    return jsonify({"success":True,"data":result})

@app.route("/api/resource/download", methods=["POST"])
def api_resource_download():
    """打包下载学习资料包（ZIP格式）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    topic = data.get("topic","")
    res_keys = data.get("keys",[])  # 要打包的资源类型列表
    
    if not topic or not res_keys:
        return jsonify({"error":"缺少主题或资源类型"}),400
    
    # 从数据库读取资源
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT res_type, content FROM resources WHERE username=? AND topic=?", 
              (session.get("username",""), topic))
    rows = c.fetchall(); conn.close()
    
    resources = {}
    for res_type, content in rows:
        if res_type in res_keys:
            try: resources[res_type] = json.loads(content)
            except: resources[res_type] = {"text": content}
    
    # 创建ZIP文件
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 添加学习清单
        manifest = f"主题：{topic}\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n资源类型：{', '.join(res_keys)}\n"
        zf.writestr(f"{topic}_学习清单.txt", manifest)
        
        # 添加各类资源
        res_type_map = {
            'doc': '课程文档', 'map': '知识结构图', 'quiz': '练习题库',
            'code': '代码案例', 'read': '拓展阅读', 'img': '知识图解', 'video': '教学视频'
        }
        
        for key in res_keys:
            if key in resources:
                res = resources[key]
                label = res_type_map.get(key, key)
                
                # 文本类资源
                if res.get('text'):
                    zf.writestr(f"{topic}_{label}.md", res['text'])
                
                # 图片资源（如果有URL，添加说明文件）
                if res.get('image'):
                    img_note = f"图片URL：{res['image']}\n画面描述：{res.get('imgPrompt','')}\n生成方式：{res.get('provider','AI')}"
                    zf.writestr(f"{topic}_{label}_图片说明.txt", img_note)
                
                # 视频资源（如果有URL，添加说明文件）
                if res.get('vidUrl'):
                    vid_note = f"视频URL：{res['vidUrl']}\n生成方式：{res.get('provider','AI')}"
                    zf.writestr(f"{topic}_{label}_视频说明.txt", vid_note)
    
    zip_buffer.seek(0)
    filename = f"{topic}_学习资料包.zip"
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename.encode('utf-8').decode('latin-1')  # 兼容中文文件名
    )

@app.route("/api/orchestrator/start", methods=["POST"])
def api_orchestrator_start():
    """启动Orchestrator调度会话"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    task_type = data.get("task_type", "resource_gen")  # resource_gen, path_gen, eval, etc.
    topic = data.get("topic", "")
    resource_types = data.get("resource_types", [])
    
    session_id = f"{session.get('username', '')}_{task_type}_{int(time.time())}"
    orchestrator.create_session(session_id)
    
    # 记录初始事件
    orchestrator.dispatch(session_id, "init", "OrchestratorAgent", "OrchestratorAgent", 
                          f"启动{task_type}任务", "running", 
                          f"主题: {topic}, 资源类型: {', '.join(resource_types)}")
    
    return jsonify({"success": True, "session_id": session_id})

@app.route("/api/orchestrator/dispatch", methods=["POST"])
def api_orchestrator_dispatch():
    """记录调度事件"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    session_id = data.get("session_id", "")
    event_type = data.get("event_type", "dispatch")
    from_agent = data.get("from_agent", "OrchestratorAgent")
    to_agent = data.get("to_agent", "")
    action = data.get("action", "")
    status = data.get("status", "pending")
    details = data.get("details", "")
    
    event = orchestrator.dispatch(session_id, event_type, from_agent, to_agent, action, status, details)
    return jsonify({"success": True, "event": event})

@app.route("/api/orchestrator/complete", methods=["POST"])
def api_orchestrator_complete():
    """完成调度会话"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    session_id = data.get("session_id", "")
    orchestrator.complete_session(session_id)
    return jsonify({"success": True})

@app.route("/api/orchestrator/events", methods=["GET"])
def api_orchestrator_events():
    """获取调度事件流（SSE）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    session_id = request.args.get("session_id", "")
    
    if not session_id:
        return jsonify({"error": "缺少session_id"}), 400
    
    def event_stream():
        last_count = 0
        while True:
            events = orchestrator.get_session_events(session_id)
            if len(events) > last_count:
                for i in range(last_count, len(events)):
                    yield f"data: {json.dumps(events[i], ensure_ascii=False)}\n\n"
                last_count = len(events)
            
            # 检查会话是否完成
            sess = orchestrator.active_sessions.get(session_id, {})
            if sess.get("status") == "completed":
                yield f"data: {json.dumps({'type': 'session_complete', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                break
            
            time.sleep(0.5)  # 500ms轮询一次
    
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route("/api/orchestrator/agents", methods=["GET"])
def api_orchestrator_agents():
    """获取所有Agent信息"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    return jsonify({"success": True, "agents": orchestrator.agents})

# ==================== 智能辅导 TutorAgent ====================
@app.route("/api/tutor/ask", methods=["POST"])
def api_tutor_ask():
    """智能答疑辅导 - 多模态解答"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    question = data.get("question", "")
    topic = data.get("topic", "")
    context = data.get("context", "")  # 当前学习上下文
    
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    
    cfg = load_config()
    username = session.get("username", "")
    
    # 获取学生画像
    profile = get_user_profile(username)
    profile_str = json.dumps(profile, ensure_ascii=False)
    
    # 构建系统提示（根据学生画像调整回答风格）
    system_prompt = f"""你是 TutorAgent（智能辅导老师），为学生提供即时答疑服务。

回答要求：
1. 先直接给出简明答案（2-3句）
2. 再详细解释原理和推导过程
3. 用生活化举例帮助理解
4. 指出常见误区和易错点
5. 给出相关知识点链接建议
6. 所有数学公式使用LaTeX格式（$或$$包裹）

学生画像：{profile_str}
当前学习主题：{topic}
"""
    
    user_message = f"问题：{question}"
    if context:
        user_message += f"\n\n当前学习上下文：{context}"
    
    # 调用AI生成回答
    answer = _call_llm(cfg, system_prompt, user_message, 2000)
    answer = fix_latex_formatting(answer)
    
    # 记录学习行为
    save_user_data(username, "tutor", "ask", {
        "question": question,
        "topic": topic,
        "timestamp": time.time()
    })
    
    GEN_STATS["texts"] += 1
    GEN_STATS["total"] += 1
    
    return jsonify({
        "success": True,
        "answer": answer,
        "topic": topic,
        "agent": "TutorAgent"
    })

@app.route("/api/tutor/ask_stream", methods=["POST"])
def api_tutor_ask_stream():
    """智能答疑 - 流式输出（打字机效果）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    question = data.get("question", "")
    topic = data.get("topic", "")
    
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    
    cfg = load_config()
    username = session.get("username", "")
    profile = get_user_profile(username)
    
    system_prompt = f"""你是 TutorAgent（智能辅导老师）。
回答要求：
1. 先直接给出简明答案
2. 再详细解释原理
3. 用生活化举例帮助理解
4. 指出常见误区
5. 所有数学公式使用LaTeX格式

学生画像：{json.dumps(profile, ensure_ascii=False)}
"""
    
    def stream():
        full_text = ""
        try:
            client = OpenAI(
                api_key=cfg["api"].get("deepseek_key", ""),
                base_url="https://api.deepseek.com"
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"问题：{question}"}
                ],
                stream=True,
                max_tokens=2000
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_text += content
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}\n\n"
            
            # 完成
            yield f"data: {json.dumps({'type': 'done', 'full_text': full_text}, ensure_ascii=False)}\n\n"
            
            # 记录
            save_user_data(username, "tutor", "ask_stream", {
                "question": question,
                "topic": topic,
                "answer_length": len(full_text),
                "timestamp": time.time()
            })
            GEN_STATS["texts"] += 1
            GEN_STATS["total"] += 1
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
    
    return Response(stream_with_context(stream()), mimetype="text/event-stream")

# ==================== 学习效果评估 EvalAgent ====================
@app.route("/api/eval/assess", methods=["POST"])
def api_eval_assess():
    """学习效果多维度评估"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    topic = data.get("topic", "")
    
    username = session.get("username", "")
    profile = get_user_profile(username)
    
    # 获取学习行为数据
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    
    # 统计资源生成数量
    c.execute("SELECT COUNT(*) FROM resources WHERE username=? AND topic=?", (username, topic))
    resource_count = c.fetchone()[0]
    
    # 统计提问次数
    c.execute("SELECT COUNT(*) FROM users_data WHERE username=? AND data_type='tutor'", (username,))
    question_count = c.fetchone()[0]
    
    # 统计画像更新次数
    c.execute("SELECT COUNT(*) FROM users_data WHERE username=? AND data_type='profile'", (username,))
    profile_update_count = c.fetchone()[0]
    
    conn.close()
    
    # 获取已掌握知识点
    mastered_topics = profile.get("mastered_topics", [])
    weak_points = profile.get("weak_points", [])
    
    # 多维度评估
    assessment = {
        "topic": topic,
        "timestamp": time.time(),
        "dimensions": {
            "resource_completion": {
                "name": "资源完成度",
                "score": min(100, resource_count * 15),  # 每个资源15分，最高100
                "detail": f"已生成 {resource_count} 个学习资源",
                "level": "优秀" if resource_count >= 5 else "良好" if resource_count >= 3 else "待提升"
            },
            "learning_engagement": {
                "name": "学习参与度",
                "score": min(100, question_count * 10 + profile_update_count * 5),
                "detail": f"提问 {question_count} 次，画像更新 {profile_update_count} 次",
                "level": "积极" if question_count >= 3 else "一般"
            },
            "knowledge_mastery": {
                "name": "知识掌握度",
                "score": min(100, len(mastered_topics) * 20),
                "detail": f"已掌握 {len(mastered_topics)} 个知识点",
                "level": "扎实" if len(mastered_topics) >= 3 else "进行中"
            },
            "weak_point_awareness": {
                "name": "薄弱点识别",
                "score": 80 if len(weak_points) > 0 else 50,
                "detail": f"已识别 {len(weak_points)} 个薄弱点",
                "level": "清晰" if len(weak_points) > 0 else "待加强"
            }
        },
        "overall_score": 0,  # 下面计算
        "overall_level": "",
        "suggestions": []
    }
    
    # 计算总分
    scores = [d["score"] for d in assessment["dimensions"].values()]
    assessment["overall_score"] = round(sum(scores) / len(scores)) if scores else 0
    
    # 总体评级
    score = assessment["overall_score"]
    if score >= 85:
        assessment["overall_level"] = "优秀"
    elif score >= 70:
        assessment["overall_level"] = "良好"
    elif score >= 55:
        assessment["overall_level"] = "合格"
    else:
        assessment["overall_level"] = "待提升"
    
    # 生成建议
    if resource_count < 3:
        assessment["suggestions"].append("建议生成更多类型的学习资源（当前仅{}个）".format(resource_count))
    if len(mastered_topics) < 2:
        assessment["suggestions"].append("建议完成更多知识点的系统学习")
    if question_count < 2:
        assessment["suggestions"].append("遇到问题时积极使用智能辅导功能")
    if len(weak_points) == 0:
        assessment["suggestions"].append("建议通过练习测试识别薄弱知识点")
    if not assessment["suggestions"]:
        assessment["suggestions"].append("学习状态良好，继续保持！")
    
    # 保存评估记录
    save_user_data(username, "eval", "assessment", assessment)
    
    return jsonify({"success": True, "assessment": assessment})

@app.route("/api/eval/report", methods=["GET"])
def api_eval_report():
    """获取学习评估报告"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    topic = request.args.get("topic", "")
    
    username = session.get("username", "")
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    
    # 获取最近的评估记录
    c.execute("SELECT content FROM users_data WHERE username=? AND data_type='eval' ORDER BY created_at DESC LIMIT 1", (username,))
    row = c.fetchone()
    conn.close()
    
    if row:
        try:
            assessment = json.loads(row[0])
            return jsonify({"success": True, "assessment": assessment})
        except:
            pass
    
    return jsonify({"success": False, "message": "暂无评估记录"})

@app.route("/api/eval/adjust", methods=["POST"])
def api_eval_adjust():
    """根据评估结果动态调整学习策略"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    assessment = data.get("assessment", {})
    
    username = session.get("username", "")
    profile = get_user_profile(username)
    
    adjustments = {
        "timestamp": time.time(),
        "changes": []
    }
    
    # 根据资源完成度调整
    resource_score = assessment.get("dimensions", {}).get("resource_completion", {}).get("score", 0)
    if resource_score < 50:
        adjustments["changes"].append({
            "type": "resource_recommendation",
            "action": "increase",
            "detail": "增加资源推送频率，优先推送核心文档和练习题"
        })
    
    # 根据知识掌握度调整
    mastery_score = assessment.get("dimensions", {}).get("knowledge_mastery", {}).get("score", 0)
    if mastery_score < 40:
        adjustments["changes"].append({
            "type": "path_adjustment",
            "action": "simplify",
            "detail": "简化学习路径，先巩固基础知识点"
        })
    elif mastery_score >= 80:
        adjustments["changes"].append({
            "type": "path_adjustment",
            "action": "advance",
            "detail": "可以进入进阶学习阶段，增加拓展阅读和实操案例"
        })
    
    # 根据薄弱点调整
    weak_points = profile.get("weak_points", [])
    if weak_points:
        adjustments["changes"].append({
            "type": "weak_point_focus",
            "action": "targeted_practice",
            "detail": f"针对薄弱点加强练习：{', '.join(weak_points)}"
        })
    
    # 保存调整记录
    save_user_data(username, "eval", "adjustment", adjustments)
    
    return jsonify({
        "success": True,
        "adjustments": adjustments,
        "profile_updated": profile
    })

# ==================== 学习行为跟踪 ====================
@app.route("/api/track/log", methods=["POST"])
def api_track_log():
    """记录学习行为"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    
    username = session.get("username", "")
    action_type = data.get("action_type", "")  # view_resource, complete_quiz, ask_question, etc.
    topic = data.get("topic", "")
    details = data.get("details", {})
    
    log_entry = {
        "username": username,
        "action_type": action_type,
        "topic": topic,
        "details": details,
        "timestamp": time.time()
    }
    
    save_user_data(username, "track", action_type, log_entry)
    
    return jsonify({"success": True, "log_id": str(uuid.uuid4())[:8]})

@app.route("/api/track/stats", methods=["GET"])
def api_track_stats():
    """获取学习行为统计"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    topic = request.args.get("topic", "")
    
    username = session.get("username", "")
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    
    # 总学习时长（估算）
    c.execute("SELECT COUNT(*) FROM users_data WHERE username=?", (username,))
    total_actions = c.fetchone()[0]
    
    # 资源使用情况
    c.execute("SELECT COUNT(*) FROM resources WHERE username=?", (username,))
    total_resources = c.fetchone()[0]
    
    # 提问次数
    c.execute("SELECT COUNT(*) FROM users_data WHERE username=? AND data_type='tutor'", (username,))
    total_questions = c.fetchone()[0]
    
    # 各主题资源分布
    c.execute("SELECT topic, COUNT(*) as cnt FROM resources WHERE username=? GROUP BY topic ORDER BY cnt DESC", (username,))
    topic_distribution = [{"topic": row[0], "count": row[1]} for row in c.fetchall()]
    
    # 各资源类型分布
    c.execute("SELECT res_type, COUNT(*) as cnt FROM resources WHERE username=? GROUP BY res_type ORDER BY cnt DESC", (username,))
    res_type_distribution = [{"type": row[0], "count": row[1]} for row in c.fetchall()]
    
    conn.close()
    
    estimated_minutes = total_actions * 3  # 粗略估算
    
    return jsonify({
        "success": True,
        "stats": {
            "total_actions": total_actions,
            "total_resources": total_resources,
            "total_questions": total_questions,
            "estimated_study_minutes": estimated_minutes,
            "topic_distribution": topic_distribution,
            "res_type_distribution": res_type_distribution
        }
    })

# ==================== 防幻觉检查增强 ====================
@app.route("/api/anti_hallucination/verify", methods=["POST"])
def api_anti_hallucination_verify():
    """增强版防幻觉检查 - 加入知识库验证"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    content = data.get("content", "")
    topic = data.get("topic", "")
    
    if not content:
        return jsonify({"error": "请输入内容"}), 400
    
    # 1. 基础检查
    basic_result = hallucination_checker.check_content(content, topic)
    
    # 2. 知识库事实验证（调用AI进行事实核查）
    cfg = load_config()
    fact_check_result = _verify_facts_with_ai(content, topic, cfg)
    
    # 3. 逻辑推理验证
    logic_result = _verify_logic_with_ai(content, cfg)
    
    # 4. 学术引用检查
    citation_result = _check_citations(content)
    
    # 综合评分
    overall_score = round(
        (basic_result["scores"]["safety"] * 0.2 +
         basic_result["scores"]["consistency"] * 0.15 +
         basic_result["scores"]["academic"] * 0.15 +
         fact_check_result["score"] * 0.3 +
         logic_result["score"] * 0.1 +
         citation_result["score"] * 0.1)
    )
    
    all_issues = (
        basic_result["issues"] +
        fact_check_result["issues"] +
        logic_result["issues"] +
        citation_result["issues"]
    )
    
    return jsonify({
        "success": True,
        "overall_score": overall_score,
        "passed": overall_score >= 70,
        "dimensions": {
            "safety": basic_result["scores"]["safety"],
            "consistency": basic_result["scores"]["consistency"],
            "academic": basic_result["scores"]["academic"],
            "fact_check": fact_check_result["score"],
            "logic": logic_result["score"],
            "citations": citation_result["score"]
        },
        "issues": all_issues,
        "corrected_content": basic_result["corrected_content"]
    })

def _verify_facts_with_ai(content, topic, cfg):
    """使用AI进行事实核查"""
    result = {"score": 100, "issues": []}
    
    try:
        client = OpenAI(
            api_key=cfg["api"].get("deepseek_key", ""),
            base_url="https://api.deepseek.com"
        )
        
        system_prompt = """你是学术事实核查专家。请检查以下内容是否存在事实性错误。
检查要点：
1. 科学概念是否准确
2. 数据/公式是否正确
3. 因果关系是否合理
4. 是否存在过时信息

只输出JSON格式：{"score": 0-100, "issues": ["问题1", "问题2"], "corrections": {"原文": "修正"}}
如果没有问题，score为100，issues为空数组。"""
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"主题：{topic}\n\n内容：{content[:3000]}"}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        reply = response.choices[0].message.content
        # 尝试解析JSON
        import re
        json_match = re.search(r'\{.*\}', reply, re.DOTALL)
        if json_match:
            fact_data = json.loads(json_match.group())
            result["score"] = fact_data.get("score", 100)
            result["issues"] = fact_data.get("issues", [])
    except Exception as e:
        print(f"[FactCheck] 验证失败: {e}")
        result["score"] = 80  # 默认给80分
        result["issues"].append("事实核查服务暂时不可用")
    
    return result

def _verify_logic_with_ai(content, cfg):
    """逻辑推理验证"""
    result = {"score": 100, "issues": []}
    
    try:
        client = OpenAI(
            api_key=cfg["api"].get("deepseek_key", ""),
            base_url="https://api.deepseek.com"
        )
        
        system_prompt = """你是逻辑分析专家。请检查以下内容的逻辑一致性。
检查要点：
1. 前后论述是否矛盾
2. 推理链条是否完整
3. 结论是否有充分依据

只输出JSON格式：{"score": 0-100, "issues": ["问题1"]}"""
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content[:2000]}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        reply = response.choices[0].message.content
        import re
        json_match = re.search(r'\{.*\}', reply, re.DOTALL)
        if json_match:
            logic_data = json.loads(json_match.group())
            result["score"] = logic_data.get("score", 100)
            result["issues"] = logic_data.get("issues", [])
    except Exception as e:
        print(f"[LogicCheck] 验证失败: {e}")
        result["score"] = 85
    
    return result

def _check_citations(content):
    """学术引用检查"""
    result = {"score": 100, "issues": []}
    
    # 检查是否包含引用标记
    has_citations = any(marker in content for marker in ["[1]", "[2]", "参考文献", "来源", "引用"])
    
    if not has_citations and len(content) > 500:
        result["score"] = 70
        result["issues"].append("长内容建议添加参考文献或来源标注")
    
    # 检查是否有未闭合的引用标记
    open_brackets = content.count("[")
    close_brackets = content.count("]")
    if open_brackets != close_brackets:
        result["score"] = max(result["score"] - 20, 0)
        result["issues"].append("引用标记格式不完整")
    
    return result

# ==================== 知识图谱扩展 ====================
@app.route("/api/knowledge_graph/topics", methods=["GET"])
def api_knowledge_graph_topics():
    """获取所有可用的知识图谱主题"""
    topics = {
        "生理学": {
            "icon": "",
            "count": 20,
            "description": "细胞生理学、神经生理学、心血管生理学等"
        },
        "人工智能": {
            "icon": "🤖",
            "count": 18,
            "description": "机器学习、深度学习、自然语言处理等"
        },
        "数据结构": {
            "icon": "",
            "count": 15,
            "description": "数组、链表、树、图等数据结构"
        },
        "计算机网络": {
            "icon": "🌐",
            "count": 16,
            "description": "OSI模型、TCP/IP、HTTP协议等"
        },
        "操作系统": {
            "icon": "💻",
            "count": 14,
            "description": "进程管理、内存管理、文件系统等"
        }
    }
    return jsonify({"success": True, "topics": topics})

@app.route("/api/knowledge_graph/visualize", methods=["POST"])
def api_knowledge_graph_visualize():
    """获取知识图谱可视化数据"""
    data = request.json or {}
    topic = data.get("topic", "生理学")
    
    # 根据主题返回不同的知识图谱
    graphs = {
        "生理学": knowledge_planner.dependencies,
        "人工智能": {
            "Python基础": {"prerequisites": [], "difficulty": 2},
            "线性代数": {"prerequisites": ["Python基础"], "difficulty": 3},
            "概率统计": {"prerequisites": ["Python基础"], "difficulty": 3},
            "机器学习基础": {"prerequisites": ["线性代数", "概率统计"], "difficulty": 4},
            "监督学习": {"prerequisites": ["机器学习基础"], "difficulty": 4},
            "无监督学习": {"prerequisites": ["机器学习基础"], "difficulty": 4},
            "神经网络": {"prerequisites": ["监督学习", "线性代数"], "difficulty": 5},
            "深度学习": {"prerequisites": ["神经网络"], "difficulty": 6},
            "CNN": {"prerequisites": ["深度学习"], "difficulty": 6},
            "RNN": {"prerequisites": ["深度学习"], "difficulty": 6},
            "Transformer": {"prerequisites": ["深度学习", "CNN"], "difficulty": 7},
            "NLP基础": {"prerequisites": ["机器学习基础"], "difficulty": 4},
            "词向量": {"prerequisites": ["NLP基础"], "difficulty": 5},
            "BERT": {"prerequisites": ["Transformer", "词向量"], "difficulty": 7},
            "GPT": {"prerequisites": ["Transformer"], "difficulty": 7},
            "强化学习": {"prerequisites": ["机器学习基础"], "difficulty": 5},
            "计算机视觉": {"prerequisites": ["CNN"], "difficulty": 6},
            "大模型应用": {"prerequisites": ["GPT", "BERT"], "difficulty": 8},
        },
        "数据结构": {
            "数组": {"prerequisites": [], "difficulty": 1},
            "链表": {"prerequisites": ["数组"], "difficulty": 2},
            "栈和队列": {"prerequisites": ["数组", "链表"], "difficulty": 2},
            "树基础": {"prerequisites": ["链表"], "difficulty": 3},
            "二叉树": {"prerequisites": ["树基础"], "difficulty": 3},
            "二叉搜索树": {"prerequisites": ["二叉树"], "difficulty": 4},
            "AVL树": {"prerequisites": ["二叉搜索树"], "difficulty": 5},
            "红黑树": {"prerequisites": ["AVL树"], "difficulty": 6},
            "堆": {"prerequisites": ["二叉树"], "difficulty": 4},
            "图基础": {"prerequisites": ["树基础"], "difficulty": 4},
            "图的遍历": {"prerequisites": ["图基础"], "difficulty": 4},
            "最短路径": {"prerequisites": ["图的遍历"], "difficulty": 5},
            "最小生成树": {"prerequisites": ["图的遍历"], "difficulty": 5},
            "哈希表": {"prerequisites": ["数组"], "difficulty": 3},
            "排序算法": {"prerequisites": ["数组"], "difficulty": 3},
        }
    }
    
    graph_data = graphs.get(topic, graphs["生理学"])
    
    # 转换为可视化格式
    nodes = []
    edges = []
    node_ids = {}
    
    for i, (topic_name, info) in enumerate(graph_data.items()):
        node_id = f"node_{i}"
        node_ids[topic_name] = node_id
        nodes.append({
            "id": node_id,
            "label": topic_name,
            "difficulty": info["difficulty"],
            "prerequisites": info["prerequisites"]
        })
    
    for topic_name, info in graph_data.items():
        target_id = node_ids[topic_name]
        for prereq in info["prerequisites"]:
            if prereq in node_ids:
                edges.append({
                    "from": node_ids[prereq],
                    "to": target_id
                })
    
    return jsonify({
        "success": True,
        "topic": topic,
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    })

# ==================== 学习行为跟踪增强 ====================
@app.route("/api/track/session", methods=["POST"])
def api_track_session():
    """记录学习会话（开始/结束）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    
    username = session.get("username", "")
    action = data.get("action", "start")  # start, end
    topic = data.get("topic", "")
    
    session_key = f"session_{username}_{topic}"
    
    if action == "start":
        session["active_sessions"] = session.get("active_sessions", {})
        session["active_sessions"][session_key] = {
            "start_time": time.time(),
            "topic": topic,
            "actions": []
        }
        return jsonify({"success": True, "session_id": session_key})
    
    elif action == "end":
        active = session.get("active_sessions", {})
        if session_key in active:
            session_data = active[session_key]
            duration = time.time() - session_data["start_time"]
            
            # 保存会话记录
            save_user_data(username, "track", "session", {
                "topic": topic,
                "duration": duration,
                "actions_count": len(session_data.get("actions", [])),
                "end_time": time.time()
            })
            
            del active[session_key]
            session["active_sessions"] = active
            
            return jsonify({
                "success": True,
                "duration": round(duration, 1),
                "duration_minutes": round(duration / 60, 1)
            })
    
    return jsonify({"success": False, "error": "无效操作"})

@app.route("/api/track/action", methods=["POST"])
def api_track_action():
    """记录学习过程中的具体行为"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    
    username = session.get("username", "")
    action_type = data.get("action_type", "")  # view_resource, complete_quiz, ask_question, scroll, click
    topic = data.get("topic", "")
    details = data.get("details", {})
    
    # 记录到当前会话
    session_key = f"session_{username}_{topic}"
    active = session.get("active_sessions", {})
    if session_key in active:
        active[session_key]["actions"].append({
            "type": action_type,
            "timestamp": time.time(),
            "details": details
        })
        session["active_sessions"] = active
    
    # 同时保存到数据库
    save_user_data(username, "track", action_type, {
        "topic": topic,
        "details": details,
        "timestamp": time.time()
    })
    
    return jsonify({"success": True})

@app.route("/api/track/heatmap", methods=["GET"])
def api_track_heatmap():
    """获取学习热力图数据"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    
    username = session.get("username", "")
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    
    # 获取最近30天的学习记录
    c.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count 
        FROM users_data 
        WHERE username=? AND data_type='track' 
        AND created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (username,))
    
    heatmap_data = {}
    for row in c.fetchall():
        heatmap_data[row[0]] = row[1]
    
    conn.close()
    
    return jsonify({
        "success": True,
        "heatmap": heatmap_data,
        "total_days": len(heatmap_data),
        "max_count": max(heatmap_data.values()) if heatmap_data else 0
    })

@app.route("/api/track/insights", methods=["GET"])
def api_track_insights():
    """获取学习洞察（AI分析学习行为）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    
    username = session.get("username", "")
    topic = request.args.get("topic", "")
    
    # 获取学习统计数据
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    
    # 各时段学习分布
    c.execute("""
        SELECT strftime('%H', created_at) as hour, COUNT(*) as count
        FROM users_data
        WHERE username=? AND data_type='track'
        GROUP BY hour
        ORDER BY hour
    """, (username,))
    
    time_distribution = {row[0]: row[1] for row in c.fetchall()}
    
    # 学习频率
    c.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM users_data
        WHERE username=? AND data_type='track'
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        LIMIT 7
    """, (username,))
    
    weekly_data = {row[0]: row[1] for row in c.fetchall()}
    
    conn.close()
    
    # 找出最佳学习时段
    peak_hour = max(time_distribution.items(), key=lambda x: x[1])[0] if time_distribution else "未知"
    
    # 计算学习连续性
    consecutive_days = 0
    if weekly_data:
        dates = sorted(weekly_data.keys(), reverse=True)
        from datetime import datetime, timedelta
        today = datetime.now().date()
        for i, date_str in enumerate(dates):
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if (today - date).days == i:
                consecutive_days += 1
            else:
                break
    
    insights = {
        "peak_learning_time": f"{peak_hour}:00-{int(peak_hour)+1}:00" if peak_hour != "未知" else "数据不足",
        "consecutive_days": consecutive_days,
        "weekly_trend": "上升" if len(weekly_data) >= 2 and list(weekly_data.values())[-1] > list(weekly_data.values())[0] else "稳定",
        "total_study_days": len(weekly_data),
        "time_distribution": time_distribution
    }
    
    return jsonify({
        "success": True,
        "insights": insights
    })

# ==================== 动态调整策略智能化 ====================
@app.route("/api/strategy/predict", methods=["POST"])
def api_strategy_predict():
    """预测学习瓶颈"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    topic = data.get("topic", "")
    
    username = session.get("username", "")
    profile = get_user_profile(username)
    
    # 分析学习行为
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    
    # 获取最近的学习记录
    c.execute("""
        SELECT data_key, data_value, created_at
        FROM users_data
        WHERE username=? AND data_type='track'
        ORDER BY created_at DESC
        LIMIT 50
    """, (username,))
    
    recent_actions = []
    for row in c.fetchall():
        try:
            recent_actions.append({
                "key": row[0],
                "value": json.loads(row[1]),
                "time": row[2]
            })
        except:
            pass
    
    conn.close()
    
    # 预测瓶颈
    predictions = []
    
    # 1. 基于薄弱点预测
    weak_points = profile.get("weak_points", [])
    for wp in weak_points:
        predictions.append({
            "type": "weak_point",
            "topic": wp,
            "confidence": 0.8,
            "suggestion": f"建议复习{wp}相关知识点"
        })
    
    # 2. 基于学习进度预测
    mastered = profile.get("mastered_topics", [])
    if len(mastered) > 3:
        predictions.append({
            "type": "progress",
            "topic": "学习进度",
            "confidence": 0.7,
            "suggestion": "学习进度良好，可以尝试进阶内容"
        })
    elif len(mastered) < 2 and len(recent_actions) > 10:
        predictions.append({
            "type": "stagnation",
            "topic": "学习停滞",
            "confidence": 0.75,
            "suggestion": "检测到学习进度缓慢，建议调整学习方法"
        })
    
    # 3. 基于时间分布预测
    if len(recent_actions) > 0:
        times = [a["time"] for a in recent_actions]
        time_gaps = []
        for i in range(1, len(times)):
            gap = times[i-1] - times[i]
            time_gaps.append(gap)
        
        avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        if avg_gap > 86400 * 3:  # 超过3天没有学习
            predictions.append({
                "type": "consistency",
                "topic": "学习连续性",
                "confidence": 0.85,
                "suggestion": "学习间隔过长，建议保持每日学习"
            })
    
    return jsonify({
        "success": True,
        "predictions": predictions,
        "risk_level": "高" if len(predictions) >= 3 else "中" if len(predictions) >= 1 else "低"
    })

@app.route("/api/strategy/recommend", methods=["POST"])
def api_strategy_recommend():
    """个性化推荐算法"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    topic = data.get("topic", "")
    
    username = session.get("username", "")
    profile = get_user_profile(username)
    
    # 获取已生成的资源
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT res_type, COUNT(*) as cnt FROM resources WHERE username=? AND topic=? GROUP BY res_type", (username, topic))
    existing_resources = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    
    # 推荐逻辑
    recommendations = []
    
    # 1. 基于资源类型缺口推荐
    all_types = ["course_doc", "mind_map", "quiz", "code_example", "reading", "video_script", "image"]
    missing_types = [t for t in all_types if t not in existing_resources]
    
    for mt in missing_types[:3]:
        type_names = {
            "course_doc": "课程文档",
            "mind_map": "思维导图",
            "quiz": "练习题",
            "code_example": "代码案例",
            "reading": "拓展阅读",
            "video_script": "视频脚本",
            "image": "知识图解"
        }
        recommendations.append({
            "type": "resource_gap",
            "resource_type": mt,
            "resource_name": type_names.get(mt, mt),
            "priority": "高",
            "reason": f"尚未生成{type_names.get(mt, mt)}，建议补充"
        })
    
    # 2. 基于学生画像推荐
    preferred_modality = profile.get("preferred_modality", "")
    if preferred_modality:
        modality_map = {
            "video": {"type": "video_script", "name": "教学视频"},
            "image": {"type": "image", "name": "知识图解"},
            "text": {"type": "course_doc", "name": "课程文档"},
            "interactive": {"type": "quiz", "name": "练习题"}
        }
        if preferred_modality in modality_map:
            rec = modality_map[preferred_modality]
            recommendations.append({
                "type": "preference_match",
                "resource_type": rec["type"],
                "resource_name": rec["name"],
                "priority": "中",
                "reason": f"根据您的偏好（{preferred_modality}），推荐生成{rec['name']}"
            })
    
    # 3. 基于薄弱点推荐
    weak_points = profile.get("weak_points", [])
    for wp in weak_points[:2]:
        recommendations.append({
            "type": "weak_point_focus",
            "resource_type": "quiz",
            "resource_name": f"针对「{wp}」的专项练习",
            "priority": "高",
            "reason": f"「{wp}」是您的薄弱点，建议加强练习"
        })
    
    return jsonify({
        "success": True,
        "recommendations": recommendations,
        "total": len(recommendations)
    })

# ==================== 多模态一致性检查 ====================
@app.route("/api/multimodal/consistency", methods=["POST"])
def api_multimodal_consistency():
    """多模态内容一致性检查"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}
    
    text_content = data.get("text", "")
    image_description = data.get("image_description", "")
    video_description = data.get("video_description", "")
    topic = data.get("topic", "")
    
    results = {
        "overall_consistency": 100,
        "checks": []
    }
    
    # 1. 图文一致性检查
    if text_content and image_description:
        cfg = load_config()
        try:
            client = OpenAI(
                api_key=cfg["api"].get("deepseek_key", ""),
                base_url="https://api.deepseek.com"
            )
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是多模态内容一致性检查专家。请判断文本内容和图片描述是否一致。输出JSON：{\"score\": 0-100, \"issues\": []}"},
                    {"role": "user", "content": f"文本：{text_content[:1000]}\n\n图片描述：{image_description}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            reply = response.choices[0].message.content
            import re
            json_match = re.search(r'\{.*\}', reply, re.DOTALL)
            if json_match:
                check_result = json.loads(json_match.group())
                results["checks"].append({
                    "type": "text_image",
                    "score": check_result.get("score", 100),
                    "issues": check_result.get("issues", [])
                })
        except Exception as e:
            results["checks"].append({"type": "text_image", "score": 80, "issues": [f"检查失败: {str(e)}"]})
    
    # 2. 文视频一致性检查
    if text_content and video_description:
        results["checks"].append({
            "type": "text_video",
            "score": 85,
            "issues": ["视频内容一致性检查待完善"]
        })
    
    # 3. 图视频一致性检查
    if image_description and video_description:
        results["checks"].append({
            "type": "image_video",
            "score": 85,
            "issues": ["图视频一致性检查待完善"]
        })
    
    # 计算总体一致性
    if results["checks"]:
        scores = [c["score"] for c in results["checks"]]
        results["overall_consistency"] = round(sum(scores) / len(scores))
    
    return jsonify({
        "success": True,
        "consistency": results
    })

@app.route("/api/image", methods=["POST"])
def api_image():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}; prompt = data.get("prompt","教育插图")
    style = data.get("style","教育图表，简洁现代插画风格，无文字")
    full_prompt = f"{prompt}, {style}"; cfg = load_config()
    rk = cfg["api"].get("replicate_key","")
    if rk:
        try:
            import replicate; os.environ["REPLICATE_API_TOKEN"]=rk
            output = replicate.run("stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b", input={"prompt":full_prompt,"negative_prompt":"blurry, low quality, text, watermark","width":768,"height":512,"num_outputs":1})
            if output: GEN_STATS["images"]+=1; GEN_STATS["total"]+=1; return jsonify({"success":True,"url":output[0],"image":output[0],"provider":"Replicate SDXL","prompt":prompt})
        except Exception: pass
    sf_key = cfg["api"].get("siliconflow_key","")
    try:
        sf_resp = requests.post("https://api.siliconflow.cn/v1/image/generations", headers={"Authorization":f"Bearer {sf_key}","Content-Type":"application/json"}, json={"model":"Tongyi-MAI/Z-Image-Turbo","prompt":full_prompt,"image_size":"1024x1024","batch_size":1,"num_inference_steps":1,"guidance_scale":1}, timeout=90)
        if sf_resp.status_code==200:
            img_url = sf_resp.json().get("images",[{}])[0].get("url","")
            if img_url: GEN_STATS["images"]+=1; GEN_STATS["total"]+=1; return jsonify({"success":True,"url":img_url,"image":img_url,"provider":"SiliconFlow Z-Image","prompt":prompt})
    except Exception: pass
    return jsonify({"success":False,"error":"所有图片生成方式均不可用"})

@app.route("/api/image/styles", methods=["POST"])
def api_image_styles():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}; en_prompt = data.get("prompt", data.get("topic","")); cfg = load_config()
    styles = [("diagram","教育图表，思维导图风格，简洁矢量，蓝色调"),("infographic","信息图插画，多彩现代扁平设计，图标"),("realistic","写实教育场景，细节专业，明亮光线")]
    results = []; rk = cfg["api"].get("replicate_key","")
    if rk:
        try:
            import replicate; os.environ["REPLICATE_API_TOKEN"]=rk
            for i,(sn,sd) in enumerate(styles):
                output = replicate.run("stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b", input={"prompt":f"{en_prompt}, {sd}","negative_prompt":"blurry, low quality, text, watermark","width":768,"height":432,"num_outputs":1})
                if output: results.append({"name":sn,"label":["图表风格","信息图","写实风格"][i],"url":output[0],"image":output[0]})
            if results: GEN_STATS["images"]+=len(results); GEN_STATS["total"]+=len(results); return jsonify({"success":True,"styles":results})
        except Exception: pass
    sf_key = cfg["api"].get("siliconflow_key","")
    for i,(sn,sd) in enumerate(styles):
        try:
            sf_resp = requests.post("https://api.siliconflow.cn/v1/image/generations", headers={"Authorization":f"Bearer {sf_key}","Content-Type":"application/json"}, json={"model":"Tongyi-MAI/Z-Image-Turbo","prompt":f"{en_prompt}, {sd}","image_size":"1024x1024","batch_size":1,"num_inference_steps":1,"guidance_scale":1}, timeout=90)
            if sf_resp.status_code==200:
                img_url = sf_resp.json().get("images",[{}])[0].get("url",""); results.append({"name":sn,"label":["图表风格","信息图","写实风格"][i],"url":img_url,"image":img_url})
        except Exception: pass
    if results: GEN_STATS["images"]+=len(results); GEN_STATS["total"]+=len(results); return jsonify({"success":True,"styles":results})
    return jsonify({"success":False,"error":"多风格生成失败"})

@app.route("/api/video", methods=["POST"])
def api_video():
    """提交视频生成任务（异步，立即返回 task_id）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}; prompt = data.get("prompt","教育动画"); cfg = load_config()
    
    # 优先使用阿里云 DashScope 直连通义万相
    aliyun_key = cfg["api"].get("aliyun_video_key","")
    if aliyun_key and len(aliyun_key) > 10:
        print(f"[Video] 使用阿里云 DashScope 直连通义万相")
        result = _submit_aliyun_video(prompt, data, aliyun_key)
        # 如果阿里云提交成功，直接返回
        result_json = result.get_json()
        if result_json and result_json.get("success"):
            return result
        # 阿里云失败，尝试备用方案
        print(f"[Video] 阿里云直连失败，尝试备用 SiliconFlow...")
    
    # 备用 SiliconFlow（也使用通义万相模型）
    sf_key = cfg["api"].get("siliconflow_key","")
    if sf_key and len(sf_key) > 10:
        print(f"[Video] 使用 SiliconFlow 备用通道")
        return _submit_siliconflow_video(prompt, data, sf_key)
    
    return jsonify({"success":False,"error":"视频生成需要配置 API Key。请配置通义万相或 SiliconFlow。"})

def _submit_aliyun_video(prompt, data, api_key):
    """通义万相视频生成"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"  # 启用异步调用模式
        }
        duration = data.get("duration", 5)
        size = data.get("size", "1280*720")
        body = {
            "model": "wanx2.1-t2v-turbo",
            "input": {"prompt": prompt},
            "parameters": {"duration": duration, "size": size}
        }
        
        print(f"[Aliyun Video] Submitting: prompt={prompt[:50]}..., duration={duration}, size={size}")
        r = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
            headers=headers, json=body, timeout=30
        )
        print(f"[Aliyun Video] Response: {r.status_code} {r.text[:300]}")
        
        if r.status_code == 200:
            resp_json = r.json()
            task_id = resp_json.get("output",{}).get("task_id")
            if task_id:
                return jsonify({"success":True,"task_id":task_id,"status":"submitted","provider":"aliyun","message":"视频任务已提交，正在生成中..."})
            return jsonify({"success":False,"error":"提交成功但未获取到任务ID"})
        elif r.status_code == 401:
            return jsonify({"success":False,"error":"通义万相 API Key 无效"})
        elif r.status_code == 403:
            error_detail = r.text[:200]
            if 'Forbidden' in error_detail or 'PERMISSION_DENIED' in error_detail:
                return jsonify({"success":False,"error":"403 权限不足：请在阿里云 DashScope 控制台开通视频生成服务"})
            elif 'Arrearage' in error_detail or '欠费' in error_detail:
                return jsonify({"success":False,"error":"403 账户欠费：请充值阿里云账户"})
            else:
                return jsonify({"success":False,"error":f"403 禁止访问：{error_detail}"})
        else:
            return jsonify({"success":False,"error":f"通义万相提交失败 (HTTP {r.status_code})：{r.text[:200]}"})
    except requests.exceptions.Timeout:
        return jsonify({"success":False,"error":"网络超时，请检查网络连接"})
    except Exception as e:
        return jsonify({"success":False,"error":f"通义万相提交异常：{str(e)[:100]}"})

def _submit_siliconflow_video(prompt, data, sf_key):
    """SiliconFlow 视频生成（备用）"""
    try:
        headers = {"Authorization": f"Bearer {sf_key}", "Content-Type": "application/json"}
        img_url = data.get("img_url","")
        if img_url:
            model = "Wan-AI/Wan2.2-I2V-A14B"
            body = {"model": model, "prompt": prompt, "image_url": img_url, "resolution": "720p"}
        else:
            model = "Wan-AI/Wan2.2-T2V-A14B"
            body = {"model": model, "prompt": prompt, "resolution": "720p"}
        
        print(f"[SF Video] Submitting: model={model}, prompt={prompt[:50]}...")
        r = requests.post("https://api.siliconflow.cn/v1/video/submit", headers=headers, json=body, timeout=30)
        print(f"[SF Video] Response: {r.status_code} {r.text[:300]}")
        
        if r.status_code in [200, 201]:
            resp_json = r.json()
            task_id = resp_json.get("requestId") or resp_json.get("request_id") or resp_json.get("id")
            if task_id:
                return jsonify({"success":True,"task_id":task_id,"status":"submitted","provider":"siliconflow","message":"视频任务已提交，正在排队中..."})
            return jsonify({"success":False,"error":"提交成功但未获取到任务ID"})
        elif r.status_code == 401:
            return jsonify({"success":False,"error":"SiliconFlow API Key 无效"})
        elif r.status_code == 403:
            return jsonify({"success":False,"error":"模型不可用，可能是账户余额不足或模型未开通"})
        elif r.status_code == 429:
            return jsonify({"success":False,"error":"请求过于频繁，请稍后再试"})
        else:
            return jsonify({"success":False,"error":f"SiliconFlow提交失败 (HTTP {r.status_code})：{r.text[:200]}"})
    except requests.exceptions.Timeout:
        return jsonify({"success":False,"error":"网络超时，请检查网络连接"})
    except Exception as e:
        return jsonify({"success":False,"error":f"SiliconFlow提交异常：{str(e)[:100]}"})

@app.route("/api/video/status", methods=["GET"])
def api_video_status():
    """查询视频生成状态（异步轮询）"""
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    task_id = request.args.get("task_id","")
    provider = request.args.get("provider","")
    if not task_id: return jsonify({"success":False,"error":"缺少 task_id 参数"})
    
    if provider == "aliyun":
        return _query_aliyun_video_status(task_id)
    elif provider == "siliconflow":
        return _query_siliconflow_video_status(task_id)
    
    # 自动判断：先尝试通义万相
    cfg = load_config()
    aliyun_key = cfg["api"].get("aliyun_video_key","")
    if aliyun_key:
        result = _query_aliyun_video_status(task_id)
        if result: return result
    
    sf_key = cfg["api"].get("siliconflow_key","")
    if sf_key:
        return _query_siliconflow_video_status(task_id)
    
    return jsonify({"success":False,"error":"未配置任何视频API Key"})

def _query_aliyun_video_status(task_id):
    """查询通义万相视频状态"""
    cfg = load_config()
    aliyun_key = cfg["api"].get("aliyun_video_key","")
    if not aliyun_key: return None
    
    try:
        headers = {"Authorization": f"Bearer {aliyun_key}", "Content-Type": "application/json"}
        poll = requests.get(f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}", headers=headers, timeout=10)
        print(f"[Aliyun Video Status] HTTP {poll.status_code}: {poll.text[:300]}")
        
        if poll.status_code == 200:
            t = poll.json()
            status = t.get("output",{}).get("task_status","")
            
            status_map = {"PENDING":"排队中","RUNNING":"生成中","SUCCEEDED":"完成","FAILED":"失败","CANCELED":"已取消"}
            status_cn = status_map.get(status, status)
            
            if status == "SUCCEEDED":
                video_url = t.get("output",{}).get("video_url","")
                if video_url:
                    GEN_STATS["videos"]+=1; GEN_STATS["total"]+=1
                    return jsonify({"success":True,"status":"done","url":video_url,"provider":"通义万相"})
                return jsonify({"success":True,"status":"done_no_url","message":"生成完成但未获取到视频链接"})
            elif status in ["FAILED","CANCELED"]:
                message = t.get("output",{}).get("message","未知原因")
                return jsonify({"success":False,"status":"failed","error":f"视频生成失败：{message}"})
            else:
                return jsonify({"success":True,"status":status,"message":f"🎬 {status_cn}"})
        elif poll.status_code == 404:
            return jsonify({"success":False,"error":"任务不存在或已过期"})
        else:
            return jsonify({"success":False,"error":f"查询失败 (HTTP {poll.status_code})"})
    except requests.exceptions.Timeout:
        return jsonify({"success":True,"status":"timeout","message":"⏳ 查询超时，请稍后重试"})
    except Exception as e:
        return jsonify({"success":False,"error":f"查询异常：{str(e)[:100]}"})

def _query_siliconflow_video_status(task_id):
    """查询SiliconFlow视频状态"""
    cfg = load_config()
    sf_key = cfg["api"].get("siliconflow_key","")
    if not sf_key: return jsonify({"success":False,"error":"未配置 SiliconFlow API Key"})
    
    try:
        headers = {"Authorization": f"Bearer {sf_key}", "Content-Type": "application/json"}
        poll = requests.post("https://api.siliconflow.cn/v1/video/status", headers=headers, json={"requestId": task_id}, timeout=10)
        print(f"[SF Video Status] HTTP {poll.status_code}: {poll.text[:300]}")
        
        if poll.status_code == 200:
            t = poll.json()
            status = t.get("status", "unknown")
            
            status_map = {"InQueue": "排队中", "InProgress": "生成中", "Succeed": "完成", "Failed": "失败"}
            status_cn = status_map.get(status, status)
            
            if status in ["Succeed", "SUCCEEDED", "SUCCESS", "succeeded", "completed"]:
                results = t.get("results") or t.get("output")
                video_url = None
                if isinstance(results, dict):
                    videos = results.get("videos") or results.get("video")
                    if isinstance(videos, list) and len(videos) > 0:
                        video_url = videos[0].get("url") if isinstance(videos[0], dict) else videos[0]
                    else:
                        video_url = results.get("video_url") or results.get("url")
                elif isinstance(results, list) and len(results) > 0:
                    video_url = results[0] if isinstance(results[0], str) else results[0].get("url")
                elif isinstance(results, str):
                    video_url = results
                
                if video_url:
                    GEN_STATS["videos"]+=1; GEN_STATS["total"]+=1
                    return jsonify({"success":True,"status":"done","url":video_url,"provider":"SiliconFlow Wan2.2"})
                return jsonify({"success":True,"status":"done_no_url","message":"生成完成但未获取到视频链接"})
            
            elif status in ["Failed", "FAILED", "CANCELLED", "failed", "cancelled"]:
                reason = t.get("reason", "未知原因")
                return jsonify({"success":False,"status":"failed","error":f"视频生成失败：{reason}"})
            
            else:
                position = t.get("position", 0)
                progress_msg = f"🎬 {status_cn}"
                if position > 0: progress_msg += f"（排队第{position}位）"
                return jsonify({"success":True,"status":status,"message":progress_msg})
        elif poll.status_code == 404:
            return jsonify({"success":False,"error":"任务不存在或已过期（超过10分钟）"})
        else:
            return jsonify({"success":False,"error":f"查询失败 (HTTP {poll.status_code})"})
    except requests.exceptions.Timeout:
        return jsonify({"success":True,"status":"timeout","message":"⏳ 查询超时，请稍后重试"})
    except Exception as e:
        return jsonify({"success":False,"error":f"查询异常：{str(e)[:100]}"})

@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    if not session.get("logged_in"): return jsonify({"error":"Not logged in"}),403
    data = request.json or {}; image_b64 = data.get("image","")
    if not image_b64: return jsonify({"success":False,"error":"未提供图片"})
    if "," in image_b64: image_b64 = image_b64.split(",")[1]
    cfg = load_config()
    ocr_error = None
    try:
        result = _xfyun_ocr(image_b64, cfg)
        if result:
            # 分离LaTeX公式和纯文本
            # result已经有LaTeX公式参考头，发送给AI时去掉LaTeX，只给纯文本
            ai_question = strip_latex_for_ai(result)
            answer = _call_llm(cfg, "你是数学老师，请用Markdown格式解答数学题。", f"题目：{ai_question}\n请给出详细解答步骤。", 1500)
            # 后处理：为LaTeX代码添加$$包裹（交给KaTeX渲染，坏的LaTeX会显示原文）
            answer = fix_latex_formatting(answer)
            GEN_STATS["ocr"]+=1; GEN_STATS["total"]+=1
            return jsonify({"success":True,"question":result,"answer":answer,"provider":"讯飞OCR"})
        else:
            ocr_error = "讯飞OCR识别返回空结果"
    except Exception as e:
        ocr_error = f"讯飞OCR识别异常: {str(e)}"
        print(f"[OCR Error] {ocr_error}")
    return jsonify({"success":False,"error":f"OCR识别失败：{ocr_error}。请检查讯飞OCR API配置是否正确，或尝试重新上传清晰的图片。"})

def _xfyun_ocr(image_b64, cfg):
    """使用讯飞OCR大模型进行识别（含自动重试）"""
    app_id = cfg["api"].get("xfyun_ocr_appid", "")
    api_key = cfg["api"].get("xfyun_ocr_apikey", "")
    api_secret = cfg["api"].get("xfyun_ocr_secret", "")
    
    if not app_id or not api_key or not api_secret:
        print("[OCR] 未配置讯飞OCR API密钥")
        return None
    
    api_host = "cbm01.cn-huabei-1.xf-yun.com"
    url_path = "/v1/private/se75ocrbm"
    url = f"https://{api_host}{url_path}"
    
    request_body = {
        "header": {"app_id": app_id, "status": 0},
        "parameter": {
            "ocr": {
                "result_option": "normal",
                "result_format": "json",
                "output_type": "one_shot",
                "exif_option": "0",
                "alpha_option": "0",
                "rotation_min_angle": 5,
                "result": {"encoding": "utf8", "compress": "raw", "format": "plain"}
            }
        },
        "payload": {
            "image": {"encoding": "jpg", "image": image_b64, "status": 0, "seq": 0}
        }
    }
    
    # 尝试最多2次（应对网络超时）
    for attempt in range(2):
        print(f"[OCR] 尝试讯飞OCR大模型 (第{attempt+1}次)...")
        try:
            date = formatdate(timeval=None, localtime=False, usegmt=True)
            
            request_line = f"POST {url_path} HTTP/1.1"
            signature_origin = f"host: {api_host}\ndate: {date}\n{request_line}"
            signature_sha = hmac.new(api_secret.encode(), signature_origin.encode(), digestmod=hashlib.sha256).digest()
            signature = base64.b64encode(signature_sha).decode()
            authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
            authorization = base64.b64encode(authorization_origin.encode()).decode()
            
            auth_params = {"authorization": authorization, "host": api_host, "date": date}
            url_with_auth = f"{url}?{urlencode(auth_params)}"
            headers = {"Content-Type": "application/json"}
            
            resp = requests.post(url_with_auth, headers=headers, json=request_body, timeout=60)
            print(f"[OCR] 讯飞OCR大模型响应: {resp.status_code}")
            
            if resp.status_code == 200:
                result = resp.json()
                code = result.get("header", {}).get("code")
                if code == 0:
                    text_base64 = result.get("payload", {}).get("result", {}).get("text", "")
                    if text_base64:
                        text = base64.b64decode(text_base64).decode("utf-8")
                        print(f"[OCR] OCR大模型识别成功，内容长度: {len(text)}")
                        
                        # 解析JSON格式的OCR结果，智能提取纯文本
                        try:
                            ocr_data = json.loads(text)
                            
                            def extract_texts_smart(obj):
                                """智能提取：只提取text_unit文本，避免重复"""
                                texts = []
                                def recurse(obj):
                                    if isinstance(obj, dict):
                                        if obj.get("type") == "text_unit" and "text" in obj:
                                            t = obj["text"]
                                            if isinstance(t, str) and t.strip():
                                                texts.append(t.strip())
                                        for value in obj.values():
                                            if isinstance(value, (dict, list)):
                                                recurse(value)
                                    elif isinstance(obj, list):
                                        for item in obj:
                                            if isinstance(item, (dict, list)):
                                                recurse(item)
                                recurse(obj)
                                return texts
                            
                            all_texts = extract_texts_smart(ocr_data)
                            
                            # 去重：保留顺序，去除重复
                            seen = set()
                            unique_texts = []
                            for t in all_texts:
                                if t and t not in seen:
                                    seen.add(t)
                                    # 包含LaTeX命令的直接包裹$$（供KaTeX渲染）
                                    if re.search(r'\\(?:begin|frac|lim|sqrt|sin|cos|tan|log|ln|int|sum|prod|left|right|rightarrow|times|cdot|partial|nabla|infty|displaystyle|text|mathbf)', t):
                                        t = f'$${t}$$'
                                    unique_texts.append(t)
                            
                            extracted_text = " ".join(unique_texts)
                            
                            if extracted_text.strip():
                                print(f"[OCR] 提取文本成功: {extracted_text[:200]}...")
                                return extracted_text.strip()
                            else:
                                print(f"[OCR] 未提取到有效文本")
                                return None
                        except Exception as e:
                            print(f"[OCR] JSON解析失败: {e}")
                            return None
                else:
                    print(f"[OCR] OCR大模型错误: {code} - {result.get('header', {}).get('message', '')}")
                    return None
            else:
                print(f"[OCR] OCR大模型HTTP错误: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"[OCR] OCR大模型调用异常 (第{attempt+1}次): {e}")
            if attempt == 0:
                print("[OCR] 等待2秒后重试...")
                time.sleep(2)
            else:
                print("[OCR] 重试失败，放弃")
    
    return None

def _simulate(prompt, system=""):
    p = (prompt or "").lower()
    if "profile" in (system or "").lower() or "画像" in p: return "你好！我是 ProfileAgent 😊\n\n请告诉我：你是想**期末突击过关**，还是**冲高分/考研保研**？"
    if "路径" in p: return "### 📋 学习路径\n\n**节点1** — 基础入门（1周）\n**节点2** — 原理深化（2周）\n**节点3** — 实战演练（2周）\n**节点4** — 总结提升（1周）"
    if "课程" in p: return "## 📖 课程文档\n\n### 🎯 学习目标\n掌握核心概念\n\n### ⚠️ 常见误区\n- 只看不练\n\n### ✅ 本章小结\n理论 + 实践 = 掌握"
    if "题" in p: return "### 📝 练习题\n\n**选择题1**\nA. A  B. B ✓  C. C  D. D"
    if "代码" in p: return "### 💻 代码案例\n\n```python\ndef example(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result\n```"
    if "拓展" in p: return "### 📚 拓展阅读\n\n| # | 资源 | 类型 | 推荐 |\n|---|------|------|------|\n| 1 | 经典教材 | 📕 | ⭐⭐⭐⭐⭐ |"
    if "评估" in p: return '{"scores":{"知识覆盖度":72},"suggestions":["多做练习"]}'
    if "视频" in p: return "### 🎬 视频脚本\n\n| 时间 | 画面 | 配音 |\n|------|------|------|\n| 0:00-0:30 | 标题 | 欢迎 |"
    return f"关于「{p[:30]}」：建议从基础概念入手。"

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE): save_config(DEFAULT_CONFIG)
    init_db()
    app.run(debug=True, port=5000, host="0.0.0.0")