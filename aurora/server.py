from __future__ import annotations

import argparse
import json
import logging
import threading
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn

from aurora.config import ExperimentConfig
from aurora.generation import generate
from aurora.inference import inference_autocast, load_model_from_checkpoint
from aurora.tokenizer import BaseTokenizer, load_tokenizer
from aurora.utils import resolve_device

MAX_REQUEST_BYTES = 1_048_576


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class GenerationService:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        tokenizer: BaseTokenizer,
        device: torch.device | str,
        max_new_tokens: int = 512,
        max_prompt_tokens: int = 4096,
        seed: int = 42,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = seed
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if max_new_tokens < 0 or max_new_tokens > self.max_new_tokens:
            raise ValueError(f"max_new_tokens must be in [0, {self.max_new_tokens}]")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(top_p, (int, float)) or not 0.0 <= float(top_p) <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if not isinstance(repetition_penalty, (int, float)) or repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if not isinstance(temperature, (int, float)) or temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")

        with self._lock:
            prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            if len(prompt_ids) > self.max_prompt_tokens:
                raise ValueError(f"prompt must not exceed {self.max_prompt_tokens} tokens")
            input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed if seed is None else seed)

            with inference_autocast(self.device, self.config.training.dtype_amp):
                tokens, _ = generate(
                    self.model,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    eos_id=self.tokenizer.eos_id,
                    generator=generator,
                )

            finish_reason = "eos" if tokens and tokens[-1] == self.tokenizer.eos_id else "length"
            return {
                "text": self.tokenizer.decode(tokens),
                "tokens": len(tokens),
                "prompt_tokens": len(prompt_ids),
                "finish_reason": finish_reason,
            }


def make_handler(
    service: GenerationService, logger: logging.Logger
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AuroraServer/0.1"
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and content_type != "application/json":
                raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected application/json")
            length_header = self.headers.get("Content-Length")
            if length_header is None:
                raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            try:
                length = int(length_header)
            except ValueError as error:
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from error
            if length < 0:
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            if length > MAX_REQUEST_BYTES:
                raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as error:
                raise RequestError(HTTPStatus.BAD_REQUEST, "invalid JSON body") from error
            if not isinstance(payload, dict):
                raise RequestError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return payload

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "architecture": service.config.model.architecture,
                        "device": str(service.device),
                    },
                )
                return
            if path == "/generate":
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "use POST /generate"},
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/generate":
                self.close_connection = True
                if path == "/health":
                    self._send_json(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        {"error": "use GET /health"},
                    )
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                payload = self._read_json()
                result = service.generate(
                    prompt=payload.get("prompt", ""),
                    max_new_tokens=payload.get("max_new_tokens", 64),
                    temperature=payload.get("temperature", 1.0),
                    top_k=payload.get("top_k", 0),
                    top_p=payload.get("top_p", 1.0),
                    repetition_penalty=payload.get("repetition_penalty", 1.0),
                    seed=payload.get("seed"),
                )
            except RequestError as error:
                self.close_connection = True
                self._send_json(error.status, {"error": str(error)})
                return
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception:
                logger.exception("Generation request failed")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal server error"},
                )
                return
            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    return Handler


def serve(service: GenerationService, host: str, port: int, logger: logging.Logger) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(service, logger))
    server.daemon_threads = True
    logger.info("Serving on http://%s:%d", host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an AURORA checkpoint over HTTP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", choices=["byte", "sentencepiece"], default="byte")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("aurora.server")
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be in [1, 65535]")
    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device=device)
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_model)
    service = GenerationService(
        model,
        config,
        tokenizer,
        device,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        seed=args.seed,
    )
    serve(service, args.host, args.port, logger)


if __name__ == "__main__":
    main()
