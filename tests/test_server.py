from __future__ import annotations

import http.client
import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from http.server import ThreadingHTTPServer

import pytest

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig
from aurora.server import MAX_REQUEST_BYTES, GenerationService, make_handler
from aurora.tokenizer import ByteTokenizer


@pytest.fixture
def service(tiny_experiment_config: ExperimentConfig) -> GenerationService:
    model_config = replace(tiny_experiment_config.model, vocab_size=260)
    model = build_model(model_config)
    return GenerationService(
        model,
        ExperimentConfig(model=model_config, training=tiny_experiment_config.training),
        ByteTokenizer(),
        device="cpu",
        max_new_tokens=8,
        max_prompt_tokens=16,
    )


@pytest.fixture
def server_url(service: GenerationService) -> Iterator[str]:
    logger = logging.getLogger("aurora.server.test")
    logger.setLevel(logging.CRITICAL)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, logger))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_service_generate(service: GenerationService) -> None:
    result = service.generate("hello", max_new_tokens=3, temperature=0.0)
    assert isinstance(result["text"], str)
    assert 1 <= result["tokens"] <= 3
    assert result["prompt_tokens"] == len("hello") + 1
    assert result["finish_reason"] in {"eos", "length"}


def test_service_validates_parameters(service: GenerationService) -> None:
    with pytest.raises(ValueError):
        service.generate("hello", max_new_tokens=9)
    with pytest.raises(ValueError):
        service.generate(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.generate("hello", temperature=-1.0)
    with pytest.raises(ValueError):
        service.generate("hello", top_p=1.5)
    with pytest.raises(ValueError):
        service.generate("hello", top_k=-1)
    with pytest.raises(ValueError):
        service.generate("hello", seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.generate("x" * 20)


def test_http_health_and_generate(server_url: str) -> None:
    with urllib.request.urlopen(f"{server_url}/health", timeout=5) as response:
        assert response.status == 200
        health = json.loads(response.read())
    assert health["status"] == "ok"
    assert health["architecture"] == "aurora"

    payload = json.dumps(
        {
            "prompt": "hello",
            "max_new_tokens": 3,
            "temperature": 0.0,
            "seed": 5,
        }
    ).encode()
    request = urllib.request.Request(
        f"{server_url}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
        result = json.loads(response.read())
    assert 1 <= result["tokens"] <= 3
    assert result["prompt_tokens"] == 6


def test_http_rejects_bad_requests(server_url: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(f"{server_url}/missing", timeout=5)
    assert missing.value.code == 404

    with pytest.raises(urllib.error.HTTPError) as method:
        urllib.request.urlopen(f"{server_url}/generate", timeout=5)
    assert method.value.code == 405

    request = urllib.request.Request(
        f"{server_url}/generate",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as invalid:
        urllib.request.urlopen(request, timeout=5)
    assert invalid.value.code == 400

    request = urllib.request.Request(
        f"{server_url}/generate",
        data=json.dumps({"prompt": "hello", "top_p": 2.0}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as parameters:
        urllib.request.urlopen(request, timeout=5)
    assert parameters.value.code == 400


def test_http_rejects_oversized_body(server_url: str) -> None:
    host = server_url.removeprefix("http://")
    connection = http.client.HTTPConnection(host, timeout=5)
    try:
        connection.request(
            "POST",
            "/generate",
            body=b"",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(MAX_REQUEST_BYTES + 1),
            },
        )
        response = connection.getresponse()
        assert response.status == 413
    finally:
        connection.close()
