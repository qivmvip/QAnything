from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)
from abc import ABC
import tiktoken
import os
from dotenv import load_dotenv
from http import HTTPStatus
from dashscope import api_key
from dashscope import Generation
from dashscope import Tokenization

from typing import Optional, List
import sys
import json
import requests
from qanything_kernel.utils.custom_log import debug_logger
sys.path.append("../../../")

load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
# OPENAI_API_MODEL_NAME = os.getenv("OPENAI_API_MODEL_NAME")
# OPENAI_API_CONTEXT_LENGTH = os.getenv("OPENAI_API_CONTEXT_LENGTH")
# if isinstance(OPENAI_API_CONTEXT_LENGTH, str) and OPENAI_API_CONTEXT_LENGTH != '':
#     OPENAI_API_CONTEXT_LENGTH = int(OPENAI_API_CONTEXT_LENGTH)
# debug_logger.info(f"OPENAI_API_BASE = {OPENAI_API_BASE}")
# debug_logger.info(f"OPENAI_API_MODEL_NAME = {OPENAI_API_MODEL_NAME}")

api_key="sk-xx"

class QwenLLM(BaseAnswer, ABC):
    model: str = None
    token_window: int = None
    max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0
    top_p: float = 1.0  # top_p must be (0,1]
    stop_words: str = None
    history: List[List[str]] = []
    history_len: int = 2

    def __init__(self, args):
        super().__init__()
        # api_key = args.qwen_api_key
        # self.model = args.openai_api_model_name
        self.token_window = int(args.openai_api_context_length)
        # debug_logger.info(f"QWEN_API_KEY = {api_key}")
        # debug_logger.info(f"QWEN_API_MODEL_NAME = {self.model}")
        # debug_logger.info(f"QWEN_API_CONTEXT_LENGTH = {self.token_window}")

    @property
    def _llm_type(self) -> str:
        return "using qwen API serve as LLM backend"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    # 定义函数 num_tokens_from_messages，该函数返回由一组消息所使用的token数
    def num_tokens_from_messages(self, messages, model=None):
        return 5

    def num_tokens_from_docs(self, docs):
        return 5

    async def _call(self, prompt: str, history: List[List[str]], streaming: bool = False) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        debug_logger.info(messages)

        streaming = False
        try:
            if streaming:
                response = Generation.call(model="qwen-max",
                                           messages=messages,
                                           result_format='message',  # 将输出设置为"message"格式
                                           stream=True,  # 设置输出方式为流式输出
                                           )
                if response.status_code == HTTPStatus.OK:
                    print(response)
                else:
                    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                debug_logger.info(f"qwen RES: {response}")
                for event in response:
                    if not isinstance(event, dict):
                        event = event.model_dump()

                    if isinstance(event['choices'], List) and len(event['choices']) > 0:
                        event_text = event["choices"][0]['delta']['content']
                        if isinstance(event_text, str) and event_text != "":
                            # debug_logger.info(f"[debug] event_text = [{event_text}]")
                            delta = {'answer': event_text}
                            yield "data: " + json.dumps(delta, ensure_ascii=False)

            else:
                response = Generation.call(model="qwen-max",
                                           messages=messages,
                                           result_format='message',  # 将输出设置为"message"格式
                                           stream=False,  # 设置输出方式为流式输出
                                           )
                if response.status_code == HTTPStatus.OK:
                    print(response)
                else:
                    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                debug_logger.info(
                    f"[debug] response.choices = [{response.output.choices}]")
                event_text = response.output.choices[0].message.content if response.output.choices else ""
                delta = {'answer': event_text}
                yield "data: " + json.dumps(delta, ensure_ascii=False)

        except Exception as e:
            debug_logger.info(f"Error calling qwen API: {e}")
            delta = {'answer': f"{e}"}
            yield "data: " + json.dumps(delta, ensure_ascii=False)

        finally:
            # debug_logger.info("[debug] try-finally")
            yield f"data: [DONE]\n\n"

    async def generatorAnswer(self, prompt: str,
                              history: List[List[str]] = [],
                              streaming: bool = False) -> AnswerResult:

        if history is None or len(history) == 0:
            history = [[]]
        debug_logger.info(f"history_len: {self.history_len}")
        debug_logger.info(f"prompt: {prompt}")
        debug_logger.info(
            f"prompt tokens: {self.num_tokens_from_messages([{'content': prompt}])}")
        debug_logger.info(f"streaming: {streaming}")

        response = self._call(prompt, history[:-1], streaming)
        complete_answer = ""
        async for response_text in response:

            if response_text:
                chunk_str = response_text[6:]
                if not chunk_str.startswith("[DONE]"):
                    chunk_js = json.loads(chunk_str)
                    complete_answer += chunk_js["answer"]

            history[-1] = [prompt, complete_answer]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response_text}
            answer_result.prompt = prompt
            yield answer_result


if __name__ == "__main__":

    llm = QwenLLM()
    streaming = True
    chat_history = []
    prompt = "你是谁"
    prompt = """参考信息：
中央纪委国家监委网站讯 据山西省纪委监委消息：山西转型综合改革示范区党工委副书记、管委会副主任董良涉嫌严重违纪违法，目前正接受山西省纪委监委纪律审查和监察调查。\\u3000\\u3000董良简历\\u3000\\u3000董良，男，汉族，1964年8月生，河南鹿邑人，在职研究生学历，邮箱random@xxx.com，联系电话131xxxxx909，1984年3月加入中国共产党，1984年8月参加工作\\u3000\\u3000历任太原经济技术开发区管委会副主任、太原武宿综合保税区专职副主任，山西转型综合改革示范区党工委委员、管委会副主任。2021年8月，任山西转型综合改革示范区党工委副书记、管委会副主任。(山西省纪委监委)
---
我的问题或指令：
帮我提取上述人物的中文名，英文名，性别，国籍，现任职位，最高学历，毕业院校，邮箱，电话
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""
    final_result = ""
    for answer_result in llm.generatorAnswer(prompt=prompt,
                                             history=chat_history,
                                             streaming=streaming):
        resp = answer_result.llm_output["answer"]
        if "DONE" not in resp:
            final_result += json.loads(resp[6:])["answer"]
        debug_logger.info(resp)

    debug_logger.info(f"final_result = {final_result}")
