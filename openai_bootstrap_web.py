import asyncio
import os
from typing import Optional, Any, Dict

from fastapi import (APIRouter,
                     FastAPI,
                     HTTPException,
                     Response,
                     Request,
                     status
                     )
import logging
from loom_core.openai_plugins.callback.bootstrap import OpenAIBootstrapBaseWeb
import json
import pprint
import tiktoken
from loom_core.openai_plugins.callback.bootstrap.openai_protocol import ChatCompletionRequest, EmbeddingsRequest, \
    ChatCompletionResponse, ModelList, EmbeddingsResponse, ChatCompletionStreamResponse, FunctionAvailable
from uvicorn import Config, Server
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse
import multiprocessing as mp
from loom_core.openai_plugins.utils import json_dumps
import threading
from zhipuai import ZhipuAI
from sse_starlette import EventSourceResponse

from generic import dictify, jsonify

logger = logging.getLogger(__name__)


class JSONResponse(StarletteJSONResponse):
    def render(self, content: Any) -> bytes:
        return json_dumps(content)


async def create_stream_chat_completion(client: ZhipuAI, chat_request: ChatCompletionRequest):
    try:
        # 将JSON字符串转换为字典
        data_dict = json.loads(jsonify(chat_request))
        tools = data_dict.get("tools", None)
        functions = data_dict.get("functions", None)

        if functions is not None:
            tools = [json.loads(jsonify(FunctionAvailable(type='function', function=f))) for f in functions]
        response = client.chat.completions.create(
            model=chat_request.model,
            messages=data_dict["messages"],
            tools=tools,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            top_p=chat_request.top_p,
            stream=chat_request.stream,
        )
        for chunk in response:
            yield jsonify(chunk)

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


class RESTFulOpenAIBootstrapBaseWeb(OpenAIBootstrapBaseWeb):
    """
    Bootstrap Server Lifecycle
    """

    def __init__(self, host: str, port: int):
        super().__init__()
        self._host = host
        self._port = port
        self._router = APIRouter()
        self._app = FastAPI()
        self._server_thread = None

    @classmethod
    def from_config(cls, cfg=None):
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 20000)

        logger.info(f"Starting openai Bootstrap Server Lifecycle at endpoint: http://{host}:{port}")
        return cls(host=host, port=port)

    def serve(self, logging_conf: Optional[dict] = None):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._router.add_api_route(
            "/v1/models",
            self.list_models,
            response_model=ModelList,
            methods=["GET"],
        )

        self._router.add_api_route(
            "/v1/embeddings",
            self.create_embeddings,
            response_model=EmbeddingsResponse,
            status_code=status.HTTP_200_OK,
            methods=["POST"],
        )
        self._router.add_api_route(
            "/v1/chat/completions",
            self.create_chat_completion,
            response_model=ChatCompletionResponse,
            status_code=status.HTTP_200_OK,
            methods=["POST"],
        )

        self._app.include_router(self._router)

        config = Config(
            app=self._app, host=self._host, port=self._port, log_config=logging_conf
        )
        server = Server(config)

        def run_server():
            server.run()

        self._server_thread = threading.Thread(target=run_server)
        self._server_thread.start()

    async def join(self):
        await self._server_thread.join()

    def set_app_event(self, started_event: mp.Event = None):
        @self._app.on_event("startup")
        async def on_startup():
            if started_event is not None:
                started_event.set()

    async def list_models(self, request: Request):
        pass

    async def create_embeddings(self, request: Request, embeddings_request: EmbeddingsRequest):
        logger.info(f"Received create_embeddings request: {pprint.pformat(embeddings_request.dict())}")
        if os.environ["API_KEY"] is None:
            authorization = request.headers.get("Authorization")
            authorization = authorization.split("Bearer ")[-1]
        else:
            authorization = os.environ["API_KEY"]
        client = ZhipuAI(api_key=authorization)
        # 判断embeddings_request.input是否为list
        input = None
        if isinstance(embeddings_request.input, list):
            tokens = embeddings_request.input
            try:
                encoding = tiktoken.encoding_for_model(embeddings_request.model)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                encoding = tiktoken.get_encoding(model)
            for i, token in enumerate(tokens):
                text = encoding.decode(token)
                input += text

        else:
            input = embeddings_request.input

        response = client.embeddings.create(
            model=embeddings_request.model,
            input=input,
        )
        return EmbeddingsResponse(**dictify(response))

    async def create_chat_completion(self, request: Request, chat_request: ChatCompletionRequest):
        logger.info(f"Received chat completion request: {pprint.pformat(chat_request.dict())}")
        if os.environ["API_KEY"] is None:
            authorization = request.headers.get("Authorization")
            authorization = authorization.split("Bearer ")[-1]
        else:
            authorization = os.environ["API_KEY"]
        client = ZhipuAI(api_key=authorization)
        if chat_request.stream:
            generator = create_stream_chat_completion(client, chat_request)
            return EventSourceResponse(generator, media_type="text/event-stream")
        else:
            data_dict = json.loads(jsonify(chat_request))
            tools = data_dict.get("tools", None)
            functions = data_dict.get("functions", None)

            if functions is not None:
                tools = [json.loads(jsonify(FunctionAvailable(type='function', function=f))) for f in functions]

            response = client.chat.completions.create(
                model=chat_request.model,
                messages=data_dict["messages"],
                tools=tools,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
                stream=chat_request.stream,
            )

            chat_response = ChatCompletionResponse(**dictify(response))
            function_call = data_dict.get("function_call", None)
            if function_call is not None:
                for res in chat_response.choices:
                    # 筛序res.message.tool_calls中type是 function的第一个数据
                    if res.message.tool_calls is not None:
                        function = next(filter(lambda x: x.type == "function", res.message.tool_calls), None)
                        if function is not None:
                            res.message.function_call = function.function

            return chat_response


def run(
        cfg: Dict, logging_conf: Optional[dict] = None,
        started_event: mp.Event = None,
):
    logging.config.dictConfig(logging_conf)  # type: ignore
    try:
        os.environ["API_KEY"] = cfg.get("api_key", None)

        import signal
        # 跳过键盘中断，使用xoscar的信号处理
        signal.signal(signal.SIGINT, lambda *_: None)
        api = RESTFulOpenAIBootstrapBaseWeb.from_config(cfg=cfg.get("run_openai_api", {}))
        api.set_app_event(started_event=started_event)
        api.serve(logging_conf=logging_conf)

        async def pool_join_thread():
            await api.join()

        asyncio.run(pool_join_thread())
    except SystemExit:
        logger.info("SystemExit raised, exiting")
        raise
