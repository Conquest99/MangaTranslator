"""OpenAI 兼容的 API 翻译器（默认接 DeepSeek）。

注册到 BallonsTranslator 的 TRANSLATORS，key 是 'DeepSeek API'，
所以 引入翻译.py / pipeline.py / GUI 都能直接按名字选中它。

密钥不写进代码，按以下顺序查找：
    1. 环境变量（默认 DEEPSEEK_API_KEY）
    2. 本地密钥文件 整合/.secrets/deepseek.key（已 gitignore）

用法：
    setx DEEPSEEK_API_KEY sk-xxx           # 方式一（推荐）
    或者把 key 单独写一行到 整合/.secrets/deepseek.key
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from ballontranslator.modules.translators.base import BaseTranslator, TRANSLATORS

SECRET_FILE = Path(__file__).resolve().parent / '.secrets' / 'deepseek.key'

SYSTEM_PROMPT = (
    '你是专业的漫画/轻小说翻译。把用户给出的日文翻译成简体中文，要求：\n'
    '1. 保留原文的语气、人称和粗俗/敬语程度，不要擅自增删代词。\n'
    '2. 保留分隔符 "##"：输入里用 ## 分隔不同的句子，输出必须用相同数量的 ## 分隔。\n'
    '3. 只输出译文，不要解释、不要加引号、不要重复原文。'
)

DEEPSEEK_PARAMS: Dict = {
    'description': 'OpenAI 兼容 API 翻译器（DeepSeek / Qwen / Gemini 等）',
    'api baseurl': {
        'value': 'https://api.deepseek.com/v1',
        'display_name': 'API Base URL',
    },
    'model': {
        'value': 'deepseek-chat',
        'display_name': 'Model',
    },
    'api key env': {
        'value': 'DEEPSEEK_API_KEY',
        'display_name': 'API Key 环境变量名',
    },
    'temperature': {
        'value': 1.0,
        'display_name': 'Temperature',
    },
    'max tokens': {
        'value': 2048,
        'display_name': 'Max Tokens',
    },
    'retry attempts': {
        'value': 3,
        'display_name': 'Retry Attempts',
    },
    'timeout': {
        'value': 90,
        'display_name': 'Timeout (s)',
    },
}


class DeepSeekTranslator(BaseTranslator):
    """把整页文本合并成一次请求，靠 '##' 分隔符对齐回各个文字块。"""

    concate_text = True
    cht_require_convert = True
    params = deepcopy(DEEPSEEK_PARAMS)

    def _setup_translator(self) -> None:
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['繁體中文'] = 'Traditional Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.lang_map['English'] = 'English'
        self._client = None

    # ---------- 参数 ----------

    @property
    def api_base(self) -> str:
        return str(self.get_param_value('api baseurl')).rstrip('/')

    @property
    def model(self) -> str:
        return str(self.get_param_value('model'))

    @property
    def timeout(self) -> float:
        return float(self.get_param_value('timeout'))

    @property
    def retry_attempts(self) -> int:
        return int(self.get_param_value('retry attempts'))

    @property
    def max_tokens(self) -> int:
        return int(self.get_param_value('max tokens'))

    @property
    def temperature(self) -> float:
        return float(self.get_param_value('temperature'))

    # ---------- 密钥 ----------

    def resolve_api_key(self) -> str:
        env_name = str(self.get_param_value('api key env') or 'DEEPSEEK_API_KEY')
        key = os.environ.get(env_name, '').strip()
        if key:
            return key
        if SECRET_FILE.exists():
            key = SECRET_FILE.read_text(encoding='utf-8').strip()
            if key:
                return key
        raise RuntimeError(
            f'没有找到 API key。请设置环境变量 {env_name}，'
            f'或把 key 写到 {SECRET_FILE}'
        )

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.resolve_api_key(),
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    def unload_model(self, empty_cache: bool = False) -> bool:
        self._client = None
        return False

    # ---------- 翻译 ----------

    def _chat(self, text: str) -> str:
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': text},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = (resp.choices[0].message.content or '').strip()
                if content:
                    return content
                last_error = RuntimeError('API 返回空内容')
            except Exception as e:  # 网络错误、限流、额度不足都走重试
                last_error = e
                self.logger.warning(
                    f'API 第 {attempt}/{self.retry_attempts} 次失败：{type(e).__name__}: {e}'
                )
        raise RuntimeError(f'API 翻译失败：{last_error}') from last_error

    def _translate(self, src_list: List[str]) -> List[str]:
        return [self._chat(src) for src in src_list]


TRANSLATORS.register_module(name='DeepSeek API', force=True, module=DeepSeekTranslator)
