import os
import sys

import yaml
import time
import httpx
import traceback

# ================= 核心修复 =================
# 彻底清理代理干扰，强制直连（完全复刻你单测成功的逻辑）
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,*"

# 添加路径
sys.path.append(r"D:\GitHubDownload\OpenRCA\rca\api_config.yaml")

# ===========================================

def load_config(config_path=r"D:\GitHubDownload\OpenRCA\rca\api_config.yaml"):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs


configs = load_config()


def OpenAI_chat_completion(messages, temperature, timeout=None):
    from openai import OpenAI

    # 获取配置，如果 config.yaml 里没写对 API_BASE，就强行兜底使用你测试成功的地址！
    current_key = configs.get("API_KEY")
    current_base = configs.get("API_BASE", "https://api.openai-proxy.org/v1")

    client = OpenAI(
        api_key=current_key,
        base_url=current_base
    )

    # allow timeout override via config or argument
    timeout_val = timeout if timeout is not None else int(configs.get("TIMEOUT", 60))
    return client.chat.completions.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature,
        timeout=timeout_val,
    ).choices[0].message.content


def Google_chat_completion(messages, temperature):
    import google.generativeai as genai
    genai.configure(
        api_key=configs["API_KEY"]
    )
    genai.GenerationConfig(temperature=temperature)
    system_instruction = messages[0]["content"] if messages[0]["role"] == "system" else None
    messages = [item for item in messages if item["role"] != "system"]
    messages = [{"role": "model" if item["role"] == "assistant" else item["role"], "parts": item["content"]} for item in
                messages]
    history = messages[:-1]
    message = messages[-1]
    return genai.GenerativeModel(
        model_name=configs["MODEL"],
        system_instruction=system_instruction
    ).start_chat(
        history=history if history != [] else None
    ).send_message(message).text


def Anthropic_chat_completion(messages, temperature):
    import anthropic
    client = anthropic.Anthropic(
        api_key=configs["API_KEY"]
    )
    return client.messages.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature
    ).content


# for 3-rd party API which is compatible with OpenAI API (with different 'API_BASE')
def AI_chat_completion(messages, temperature):
    from openai import OpenAI

    client = OpenAI(
        api_key=configs["API_KEY"],
        base_url=configs["API_BASE"]
    )
    timeout_val = int(configs.get("TIMEOUT", 60))
    return client.chat.completions.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature,
        timeout=timeout_val,
    ).choices[0].message.content


def get_chat_completion(messages, temperature=0.0, timeout=None, max_retries=5):
    """Send messages to configured backend with retries and backoff.

    timeout: per-request timeout in seconds (overrides config TIMEOUT).
    max_retries: total attempts including initial try.
    """

    def send_request():
        if configs.get("SOURCE") == "AI":
            return AI_chat_completion(messages, temperature)
        elif configs.get("SOURCE") == "OpenAI":
            return OpenAI_chat_completion(messages, temperature, timeout=timeout)
        elif configs.get("SOURCE") == "Google":
            return Google_chat_completion(messages, temperature)
        elif configs.get("SOURCE") == "Anthropic":
            return Anthropic_chat_completion(messages, temperature)
        else:
            raise ValueError("Invalid SOURCE in api_config file.")

    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            return send_request()
        except Exception as e:
            # Log stack for debugging
            print(f"Attempt {attempt} failed with error: {e}")
            traceback.print_exc()
            s = str(e).lower()
            # Rate limit handling: short wait
            if '429' in s or 'rate limit' in s:
                wait = 1
                print(f"Rate limit (429). Waiting for {wait} seconds before retrying...")
                time.sleep(wait)
                continue
            # Connection/timeout errors: exponential backoff
            if 'timed out' in s or 'connecttimeout' in s or isinstance(e, (httpx.TimeoutException,)):
                if attempt == max_retries:
                    print("Maximum retries reached, raising timeout error.")
                    raise
                print(f"Network/timeout error. Backing off for {backoff} seconds before retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            # other errors: do not retry
            raise