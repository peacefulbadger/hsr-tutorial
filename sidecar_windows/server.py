"""Windows-side capture/actuator sidecar.

Two endpoints carry the whole contract:

    GET  /frame   -> image bytes (JPEG by default, PNG on request) + timing
                     headers (X-Frame-Seq, X-Capture-Epoch-Ms, X-Response-Epoch-Ms)
    POST /action  -> dispatch a tap / swipe / back / wait through adb

The sidecar holds no game knowledge. It is a remote-controlled camera and finger.

Run:
    py -m sidecar_windows.server --config configs\sidecar.toml
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adb import AdbClient, AdbConfig, AdbError
from .capture import CaptureLoop, FrameFormat

VERSION = "0.1.0"

# Coordinate space the orchestrator speaks: the emulator's current framebuffer.
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720


class ActionRequest(BaseModel):
    """One discrete input event.

    Coordinates are in framebuffer pixels (1280x720 for the pinned HSR client),
    which is also the space `input tap` uses on this emulator build -- verified
    empirically in Phase 0.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tap", "swipe", "back", "wait"]
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    duration_ms: int = Field(default=100, ge=0, le=10_000)
    label: str | None = None

    @model_validator(mode="after")
    def _check_fields(self) -> "ActionRequest":
        if self.type == "tap" and (self.x is None or self.y is None):
            raise ValueError("tap requires x and y")
        if self.type == "swipe" and None in (self.x, self.y, self.x2, self.y2):
            raise ValueError("swipe requires x, y, x2, y2")
        for name in ("x", "y", "x2", "y2"):
            value = getattr(self, name)
            if value is not None and not (0 <= value < 8192):
                raise ValueError(f"{name} out of plausible range: {value}")
        return self


class ActionResponse(BaseModel):
    ok: bool
    type: str
    dispatched_epoch_ms: int
    dispatch_ms: float
    frame_seq_at_dispatch: int | None = None


class SidecarState:
    def __init__(self, adb: AdbClient, capture: CaptureLoop, config: dict) -> None:
        self.adb = adb
        self.capture = capture
        self.config = config
        self.actions_dispatched = 0



def create_app(
    adb: AdbClient | None = None,
    *,
    target_hz: float = 6.0,
    frame_format: FrameFormat = "jpeg",
    jpeg_quality: int = 80,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = adb or AdbClient(AdbConfig())
        loop = CaptureLoop(
            client, target_hz=target_hz, frame_format=frame_format, jpeg_quality=jpeg_quality
        )
        loop.start()
        # Don't accept traffic before there is something to serve; a client that
        # connects during startup should get a frame, not a 503.
        loop.wait_for_frame(timeout_s=3.0)
        app.state.sidecar = SidecarState(
            client,
            loop,
            {"target_hz": target_hz, "frame_format": frame_format, "jpeg_quality": jpeg_quality},
        )
        try:
            yield
        finally:
            loop.stop()

    app = FastAPI(title="HSR capture/actuator sidecar", version=VERSION, lifespan=lifespan)

    def state(request: Request) -> SidecarState:
        return request.app.state.sidecar

    @app.get("/health")
    def health(request: Request) -> dict:
        sc = state(request)
        capture_health = sc.capture.health()
        return {
            "ok": capture_health["frames_captured"] > 0 and capture_health["last_frame_age_ms"] is not None,
            "version": VERSION,
            "adb_path": sc.adb.adb_path,
            "device_serial": sc.adb.serial,
            "actions_dispatched": sc.actions_dispatched,
            "capture": capture_health,
        }

    @app.get("/frame")
    def get_frame(request: Request) -> Response:
        sc = state(request)
        frame = sc.capture.latest()
        if frame is None:
            raise HTTPException(status_code=503, detail="no frame captured yet")
        response_epoch_ms = int(time.time() * 1000)
        return Response(
            content=frame.data,
            media_type=frame.content_type,
            headers={
                "X-Frame-Seq": str(frame.seq),
                "X-Content-Format": frame.content_type,
                "X-Frame-Width": str(frame.width),
                "X-Frame-Height": str(frame.height),
                "X-Capture-Epoch-Ms": str(frame.capture_epoch_ms),
                "X-Response-Epoch-Ms": str(response_epoch_ms),
                "X-Server-Age-Ms": f"{response_epoch_ms - frame.capture_epoch_ms}",
                "X-Capture-Duration-Ms": f"{frame.capture_duration_ms:.1f}",
                "Cache-Control": "no-store",
            },
        )

    @app.get("/frame/meta")
    def get_frame_meta(request: Request) -> dict:
        """Metadata only -- used by tests and by the LAN probe."""
        sc = state(request)
        frame = sc.capture.latest()
        if frame is None:
            raise HTTPException(status_code=503, detail="no frame captured yet")
        now_ms = int(time.time() * 1000)
        return {
            "seq": frame.seq,
            "width": frame.width,
            "height": frame.height,
            "bytes": len(frame.data),
            "capture_epoch_ms": frame.capture_epoch_ms,
            "response_epoch_ms": now_ms,
            "server_age_ms": now_ms - frame.capture_epoch_ms,
            "capture_duration_ms": round(frame.capture_duration_ms, 1),
        }

    @app.post("/action", response_model=ActionResponse)
    def post_action(request: Request, action: ActionRequest) -> ActionResponse:
        sc = state(request)
        started = time.perf_counter()
        dispatched_epoch_ms = int(time.time() * 1000)
        try:
            if action.type == "tap":
                sc.adb.tap(action.x, action.y, action.duration_ms)  # type: ignore[arg-type]
            elif action.type == "swipe":
                sc.adb.swipe(action.x, action.y, action.x2, action.y2, action.duration_ms)  # type: ignore[arg-type]
            elif action.type == "back":
                sc.adb.back()
            else:  # wait
                time.sleep(action.duration_ms / 1000.0)
        except AdbError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        dispatch_ms = (time.perf_counter() - started) * 1000.0
        sc.actions_dispatched += 1
        frame = sc.capture.latest()
        return ActionResponse(
            ok=True,
            type=action.type,
            dispatched_epoch_ms=dispatched_epoch_ms,
            dispatch_ms=round(dispatch_ms, 1),
            frame_seq_at_dispatch=frame.seq if frame else None,
        )

    @app.get("/stats")
    def stats(request: Request) -> dict:
        sc = state(request)
        return {
            "capture": sc.capture.health(),
            "actions_dispatched": sc.actions_dispatched,
        }

    return app


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    import tomllib

    with path.open("rb") as fh:
        return tomllib.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HSR capture/actuator sidecar")
    parser.add_argument("--host", default=None, help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8765)")
    parser.add_argument("--target-hz", type=float, default=None, help="capture rate (default 6.0)")
    parser.add_argument(
        "--frame-format", choices=("jpeg", "png"), default=None,
        help="image encoding: jpeg (fast, default) or png (plan-literal, slower)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG quality 1-100 (default 80)")
    parser.add_argument("--adb-path", default=None, help="path to adb.exe")
    parser.add_argument("--serial", default=None, help="adb serial, e.g. 127.0.0.1:16384")
    parser.add_argument("--config", type=Path, default=None, help="TOML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    section = config.get("sidecar", {})
    host = args.host or section.get("host", "0.0.0.0")
    port = args.port or int(section.get("port", 8765))
    target_hz = args.target_hz or float(section.get("target_hz", 6.0))
    frame_format = args.frame_format or section.get("frame_format", "jpeg")
    jpeg_quality = args.jpeg_quality or int(section.get("jpeg_quality", 80))
    adb_path = args.adb_path or section.get("adb_path", "")
    serial = args.serial or section.get("serial", "")

    import uvicorn

    app = create_app(
        AdbClient(AdbConfig(adb_path=adb_path, serial=serial)),
        target_hz=target_hz,
        frame_format=frame_format,
        jpeg_quality=jpeg_quality,
    )
    print(json.dumps({
        "event": "starting", "host": host, "port": port,
        "target_hz": target_hz, "frame_format": frame_format,
    }))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
