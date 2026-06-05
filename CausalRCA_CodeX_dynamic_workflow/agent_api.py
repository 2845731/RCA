import os
from openai import OpenAI

# ================= 核心修复 =================
# 在导入或初始化 OpenAI 之前，强制设置不走代理
# 这样脚本就会直接连接本地端口，而不是绕道 Squid 代理
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
# ===========================================

# 1. 配置客户端
# 建议将 localhost 改为 127.0.0.1，避免 IPv6 (::1) 解析问题
client = OpenAI(base_url="https://token-plan-cn.xiaomimimo.com/v1", api_key="YOUR_API_KEY_HERE")

print("正在尝试连接模型...")

try:
    # 2. 发送请求
    response = client.chat.completions.create(
        model="mimo-v2.5-pro",  # 你的模型路径
        messages=[{"role": "user", "content": "请问你是什么模型，千问的什么类型,参数是多大"}],
        temperature=0.7
    )
    # 3. 打印结果
    print(response.choices[0].message.content)

except Exception as e:
    print(f"❌ 依然报错: {e}")
    print("建议检查：\n1. vLLM 是否真的在运行？(ps -ef | grep vllm)\n2. 端口是否被占用？")