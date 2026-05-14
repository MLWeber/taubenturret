# taubenturret - An automated, computer vision driven water turret system.
# Copyright (C) 2026 Michael Weber
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Simple webserver to manage recorded video clips."""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from taubenturret import config

logger = logging.getLogger(__name__)
security = HTTPBasic()


class StreamProvider(Protocol):
    """Defines the required interface for serving a livestream."""

    def add_viewer(self) -> None: ...

    def remove_viewer(self) -> None: ...

    def get_stream_jpeg(self) -> bytes | None: ...


class FireRequest(BaseModel):
    """Payload for manual firing requests via the web interface."""

    x: float
    y: float


def verify_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:  # noqa: B008
    """Verify HTTP Basic Authentication headers using timing-safe comparisons."""
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"), config.WEBSERVER_USERNAME.encode("utf8")
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"), config.WEBSERVER_PASSWORD.encode("utf8")
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def create_app(stream_provider: StreamProvider, fire_callback: Callable[[float, float], None] | None = None) -> FastAPI:  # noqa: C901
    """Factory pattern to initialize and configure the FastAPI application."""
    app = FastAPI(title="TaubenTurret UI")

    # Cache the HTML template in memory on startup to avoid hitting the SD card on every request
    template_path = Path(__file__).parent / "index.html"
    try:
        with open(template_path, encoding="utf-8") as file:
            index_template = file.read()
    except FileNotFoundError:
        msg = "web/index.html not found. UI may fail to load."
        logger.exception(msg)
        index_template = "<html><body><h1>Error: index.html not found</h1></body></html>"

    def build_gallery_html() -> str:
        """Helper to generate the HTML for the video gallery."""
        record_dir = Path(config.RECORD_DIRECTORY)
        record_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in record_dir.iterdir() if f.is_file() and f.suffix in {".mp4", ".h264"}]
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        by_day = defaultdict(list)
        for f in files:
            mtime = f.stat().st_mtime
            day_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            by_day[day_str].append((f.name, f.stat().st_size))

        gallery_parts = []

        if not files:
            gallery_parts.append("<p>No video clips found in the recordings directory.</p>")
        else:
            for day_str, day_files in by_day.items():
                gallery_parts.append("<div class='day-group'>")
                gallery_parts.append("<div class='day-header'>")
                gallery_parts.append(
                    f"<input type='checkbox' autocomplete='off' onchange='toggleDay(\"{day_str}\", this.checked)'>"
                )
                gallery_parts.append(f"<span>{day_str} ({len(day_files)} videos)</span>")
                gallery_parts.append("</div><div class='gallery-grid'>")
                for f_name, size in day_files:
                    file_size_mb = size / (1024 * 1024)
                    gallery_parts.append("<div class='gallery-item'>")
                    gallery_parts.append(
                        f"<input type='checkbox' autocomplete='off' class='video-cb day-{day_str}' "
                        f"value='{f_name}' onchange='updateDeleteButton()'>"
                    )
                    gallery_parts.append(
                        f"<img src='/thumb/{f_name}' alt='thumbnail' onclick='loadVideo(\"{f_name}\")'>"
                    )
                    gallery_parts.append(f"<div class='size-label'>{f_name}<br>({file_size_mb:.2f} MB)</div>")
                    gallery_parts.append("</div>")
                gallery_parts.append("</div></div>")

        return "".join(gallery_parts)

    @app.get("/", response_class=HTMLResponse)
    def serve_index(_: str = Depends(verify_auth)) -> HTMLResponse:
        """Generate and serve the HTML index page dynamically."""
        final_html = index_template.replace("{{ GALLERY_HTML }}", build_gallery_html())
        final_html = final_html.replace("{{ FIRE_COOLDOWN_MS }}", str(int(config.WG_FIRE_COOLDOWN * 1000)))
        return HTMLResponse(content=final_html)

    @app.get("/api/gallery", response_class=HTMLResponse)
    def serve_gallery(_: str = Depends(verify_auth)) -> HTMLResponse:
        """Serve just the gallery HTML for dynamic updates."""
        return HTMLResponse(content=build_gallery_html())

    @app.get("/videos/{filename}", response_class=FileResponse)
    def serve_video(filename: str, _: str = Depends(verify_auth)) -> FileResponse:
        """Serve a requested video file (supports HTML Range requests for video scrubbing)."""
        filepath = Path(config.RECORD_DIRECTORY) / Path(filename).name
        if not filepath.exists() or filepath.suffix not in {".mp4", ".h264"}:
            raise HTTPException(status_code=404, detail="Video Not Found")
        return FileResponse(filepath, media_type="video/mp4" if filepath.suffix == ".mp4" else "video/H264")

    @app.get("/thumb/{filename}", response_class=FileResponse)
    def serve_thumbnail(filename: str, _: str = Depends(verify_auth)) -> FileResponse:
        """Serve the first frame of the video as a JPEG thumbnail."""
        filepath = Path(config.RECORD_DIRECTORY) / f"{Path(filename).stem}.jpg"
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Thumbnail Not Found")
        return FileResponse(filepath, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.delete("/videos/{filename}")
    def delete_video(filename: str, _: str = Depends(verify_auth)) -> dict[str, str]:
        """Delete a requested video file and its thumbnail."""
        filepath = Path(config.RECORD_DIRECTORY) / Path(filename).name
        if filepath.exists() and filepath.suffix in {".mp4", ".h264"}:
            filepath.unlink()
            filepath.with_suffix(".jpg").unlink(missing_ok=True)
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Video Not Found")

    @app.post("/api/fire_manual")
    def fire_manual(payload: FireRequest, _: str = Depends(verify_auth)) -> dict[str, str]:
        if not fire_callback:
            raise HTTPException(status_code=501, detail="Manual firing not configured")
        fire_callback(payload.x, payload.y)
        return {"status": "fired"}

    @app.get("/stream")
    async def serve_stream(_: str = Depends(verify_auth)) -> StreamingResponse:
        """Serve an async continuous MJPEG stream of the camera."""
        if not stream_provider:
            raise HTTPException(status_code=501, detail="Camera streaming not available")

        async def event_generator() -> AsyncGenerator[bytes, None]:
            stream_provider.add_viewer()
            last_jpeg: bytes | None = None
            try:
                while True:
                    jpeg = stream_provider.get_stream_jpeg()
                    if jpeg and jpeg is not last_jpeg:
                        yield (
                            b"--FRAME\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                        )
                        last_jpeg = jpeg
                    await asyncio.sleep(0.03)  # 30Hz polling is plenty since stream is limited to 10 FPS
            except asyncio.CancelledError:
                pass  # Client disconnected gracefully
            finally:
                stream_provider.remove_viewer()

        return StreamingResponse(
            event_generator(),
            media_type="multipart/x-mixed-replace; boundary=FRAME",
            headers={"Age": "0", "Cache-Control": "no-cache, private"},
        )

    return app


def start_webserver(
    stream_provider: StreamProvider, fire_callback: Callable[[float, float], None] | None = None
) -> uvicorn.Server:
    """Start the webserver in a background daemon thread."""
    app = create_app(stream_provider, fire_callback)

    uvicorn_config = uvicorn.Config(
        app,
        host=config.WEBSERVER_HOST,
        port=config.WEBSERVER_PORT,
        log_level="warning",  # Suppresses Uvicorn access logs to keep the console clean
    )
    server = uvicorn.Server(uvicorn_config)

    logger.info(f"Starting FastAPI webserver on http://{config.WEBSERVER_HOST}:{config.WEBSERVER_PORT}")
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    return server
