"""Serve Pink Trumpet and its local phoneme scoring endpoint."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / ".cache" / "phoneme-discriminator.joblib"
os.environ.setdefault("SCIKIT_LEARN_DATA", str(ROOT / ".cache" / "scikit_learn_data"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
_model: Any = None
_model_lock = Lock()


def get_model() -> Any:
    global _model
    with _model_lock:
        if _model is None:
            from ml.phoneme_discriminator import PhonemeDiscriminator
            _model = PhonemeDiscriminator.load_or_train(MODEL_PATH)
    return _model


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._json(HTTPStatus.OK, {
                "ok": True,
                "modelReady": MODEL_PATH.exists(),
                "phonemes": ["aa", "ao", "dcl", "iy", "sh"],
            })
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/score":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 2_000_000:
                raise ValueError("Request body must be between 1 byte and 2 MB.")
            payload = json.loads(self.rfile.read(content_length))
            phoneme = str(payload["phoneme"])
            sample_rate = int(payload["sampleRate"])
            samples = payload["samples"]
            if not isinstance(samples, list):
                raise ValueError("samples must be a JSON array.")
            result = get_model().score(samples, sample_rate, phoneme)
            self._json(HTTPStatus.OK, result)
        except (KeyError, TypeError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except ModuleNotFoundError:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": "ML dependencies are not installed. Run: python3 -m pip install -r requirements-ml.txt",
            })
        except Exception as error:
            self.log_error("scoring failed: %s", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Phoneme scoring failed."})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--prepare", action="store_true", help="Download the dataset and train the cached discriminator.")
    args = parser.parse_args()
    if args.prepare:
        model = get_model()
        print(json.dumps({
            "model": str(MODEL_PATH),
            "validationAccuracy": model.report.accuracy,
            "samples": model.report.samples,
            "phonemes": model.report.phonemes,
        }, indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Pink Trumpet running at http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
