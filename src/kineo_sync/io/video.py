from __future__ import annotations

import functools
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import av.logging
import numpy as np

# Route FFmpeg logs to libav's native C callback instead of PyAV's Python one. The Python
# callback grabs the GIL from FFmpeg's decode worker threads (thread_type="AUTO"); on
# container teardown the main thread holds the GIL inside avcodec_free_context while waiting
# for those workers to join — they block on PyGILState_Ensure → deadlock. Restoring the
# default callback removes the only place a worker thread needs the GIL, so close never hangs.
av.logging.restore_default_callback()
# Silence the per-frame libswscale spam on the CPU/software-decode path ("deprecated pixel format
# used" — the stream is full-range YUVJ420P, which swscale flags on YUV→RGB). It is harmless colour-
# range pedantry and floods stderr; keep ERROR and above so real decode failures still surface.
av.logging.set_level(av.logging.ERROR)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _frame_converter():
    """uint8 NVDEC surface (CHW) → owned float32 [0, 1]. The ``.float()`` also copies the decoded-ahead
    surface into owned memory before the decoder recycles it.

    Fused via ``torch.compile`` (float+div → one kernel, ~2× the op) — the frame shape is fixed, so it
    compiles once with no recompiles. ROBUST: if compile is unavailable (e.g. no matching triton) or
    fails at runtime, it permanently falls back to eager. Disable with KINEO_COMPILE_CONVERT=0.
    """
    import os
    import torch

    def _eager(t):
        return t.float().div_(255.0)

    if os.environ.get("KINEO_COMPILE_CONVERT", "1") in ("0", "false", ""):
        return _eager
    try:
        compiled = torch.compile(_eager)
    except Exception:
        logger.warning("torch.compile frame-converter unavailable; using eager .float()/255")
        return _eager

    state = {"fn": compiled, "ok": True}

    def _robust(t):
        if state["ok"]:
            try:
                return state["fn"](t)
            except Exception:
                logger.warning("compiled frame-converter failed at runtime; falling back to eager")
                state["ok"] = False
        return _eager(t)

    return _robust


@functools.lru_cache(maxsize=1)
def _log_ffprobe_info() -> None:
    path = shutil.which("ffprobe") or "ffprobe"
    try:
        out = subprocess.check_output([path, "-version"], stderr=subprocess.STDOUT, text=True)
        version_line = out.splitlines()[0] if out else "unknown"
    except Exception:
        version_line = "unavailable"
    logger.info("ffprobe: %s  (%s)", version_line, path)


@dataclass
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    n_frames: int = 0


def get_video_info(path: Path | str) -> VideoInfo:
    path = Path(path)
    _log_ffprobe_info()
    t0 = time.perf_counter()
    logger.info("get_video_info: opening %s", path.name)

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream found in {path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    r_frame_rate = video.get("r_frame_rate", "30/1")
    fps = float(Fraction(r_frame_rate))

    n_frames = 0
    nb_frames = video.get("nb_frames")
    if nb_frames:
        try:
            n_frames = int(nb_frames)
        except (ValueError, TypeError):
            pass

    duration = 0.0
    for key in ("duration",):
        if key in video:
            try:
                duration = float(video[key])
                break
            except (ValueError, TypeError):
                pass

    if duration == 0.0 and n_frames > 0:
        duration = n_frames / fps
    elif duration == 0.0:
        # Fall back to frame_count / fps
        if nb_frames:
            try:
                duration = int(nb_frames) / fps
            except (ValueError, TypeError):
                pass

    if n_frames == 0 and duration > 0 and fps > 0:
        n_frames = int(round(duration * fps))

    width = int(video.get("width", 0))
    height = int(video.get("height", 0))

    info = VideoInfo(
        path=path,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        n_frames=n_frames,
    )
    logger.info(
        "get_video_info: done in %.3fs  duration=%.2fs  %dx%d  %.3ffps  audio=%s",
        time.perf_counter() - t0, duration, width, height, fps, has_audio,
    )
    return info


def get_display_size(width: int, height: int, max_edge: int = 1920) -> tuple[int, int]:
    largest = max(width, height)
    if largest <= max_edge:
        return width, height
    scale = max_edge / largest
    return int(width * scale), int(height * scale)


class VideoDecoder:
    """Persistent stateful decoder backed by PyAV (FFmpeg).

    Uses CUDA hardware decoding when available; falls back to multithreaded
    software decoding transparently.
    """

    def __init__(
        self,
        path: Path | str,
        display_size: tuple[int, int] | None = None,
        try_hw_accel: bool = True,
    ) -> None:
        self._path = Path(path)
        self._display_size = display_size
        self._try_hw_accel = try_hw_accel

        self._container: av.container.InputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._decode_iter = None
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._hw_active: bool = False
        self._hw_fallback_done: bool = False

        self._open(try_hw_accel=try_hw_accel)

    def _open(self, try_hw_accel: bool) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
            self._stream = None
            self._decode_iter = None

        options: dict[str, str] = {}
        if try_hw_accel:
            options["hwaccel"] = "cuda"

        try:
            container = av.open(str(self._path), options=options)
        except av.FFmpegError:
            if try_hw_accel:
                logger.warning(
                    "VideoDecoder: CUDA open failed for %s, retrying without hwaccel",
                    self._path.name,
                )
                container = av.open(str(self._path))
                try_hw_accel = False
            else:
                raise

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        fps_frac = stream.average_rate or stream.guessed_rate
        self._fps = float(fps_frac) if fps_frac else 30.0
        self._hw_active = try_hw_accel
        self._container = container
        self._stream = stream
        self._decode_iter = container.decode(stream)

        w = stream.width or 0
        h = stream.height or 0
        logger.info(
            "VideoDecoder opened %s  hw=%s  fps=%.3f  size=%dx%d",
            self._path.name, self._hw_active, self._fps, w, h,
        )

    def seek(self, timestamp_s: float) -> None:
        ts = max(0.0, timestamp_s)
        ts_av = int(ts / float(self._stream.time_base))
        self._container.seek(ts_av, stream=self._stream)
        # Must reset the decode iterator after every seek.
        self._decode_iter = self._container.decode(self._stream)
        self._frame_count = int(round(ts * self._fps))
        logger.debug("VideoDecoder.seek  %s  ts=%.4fs", self._path.name, timestamp_s)

    def _decode_raw(self) -> tuple[av.VideoFrame | None, float]:
        """Decode next frame without reformatting. Returns (av.VideoFrame, pts_s) or (None, -1)."""
        if self._decode_iter is None:
            return None, -1.0
        try:
            frame = next(self._decode_iter)
        except StopIteration:
            logger.debug("VideoDecoder.decode_next: EOF  %s", self._path.name)
            return None, -1.0
        except av.FFmpegError as exc:
            if self._hw_active and not self._hw_fallback_done:
                logger.warning(
                    "VideoDecoder: HW decode error for %s (%s), falling back to CPU",
                    self._path.name, exc,
                )
                self._hw_fallback_done = True
                ts = self._frame_count / max(self._fps, 1.0)
                self._open(try_hw_accel=False)
                self.seek(ts)
                return self._decode_raw()
            logger.error("VideoDecoder.decode_next FFmpegError: %s", exc)
            return None, -1.0

        pts_s = (
            float(frame.pts * self._stream.time_base)
            if frame.pts is not None
            else self._frame_count / max(self._fps, 1.0)
        )
        self._frame_count += 1
        return frame, pts_s

    def _reformat(self, frame: av.VideoFrame) -> np.ndarray:
        if self._display_size is not None:
            tw, th = self._display_size
            frame = frame.reformat(width=tw, height=th, format="rgb24")
        else:
            frame = frame.reformat(format="rgb24")
        return frame.to_ndarray()

    def decode_next(self) -> tuple[np.ndarray | None, float]:
        """Return (rgb_frame, pts_s) for the next frame, or (None, -1.0) at EOF."""
        frame, pts_s = self._decode_raw()
        if frame is None:
            return None, pts_s
        return self._reformat(frame), pts_s

    def skip_frame(self) -> bool:
        """Advance by one frame without reformatting. Returns False at EOF."""
        frame, _ = self._decode_raw()
        return frame is not None

    def advance_to(self, timestamp_s: float) -> tuple[np.ndarray | None, float]:
        """Advance sequentially to timestamp_s, skipping intermediate frames without reformatting.

        Cheaper than calling decode_next() in a loop: only the target frame pays the
        YUV→RGB reformat and GPU→CPU copy cost.
        """
        target_s = timestamp_s - 0.5 / max(self._fps, 1.0)
        last_frame: av.VideoFrame | None = None
        last_pts = -1.0
        while True:
            frame, pts_s = self._decode_raw()
            if frame is None:
                if last_frame is None:
                    return None, -1.0
                return self._reformat(last_frame), last_pts
            last_frame, last_pts = frame, pts_s
            if pts_s >= target_s:
                return self._reformat(frame), pts_s

    def decode_at(self, timestamp_s: float) -> tuple[np.ndarray | None, float]:
        """Seek to timestamp then advance without reformatting intermediate frames."""
        self.seek(timestamp_s)
        return self.advance_to(timestamp_s)

    @property
    def current_time_s(self) -> float:
        return self._frame_count / max(self._fps, 1.0)

    @property
    def hw_active(self) -> bool:
        return self._hw_active

    def close(self) -> None:
        self._decode_iter = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def __enter__(self) -> VideoDecoder:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class VideoGroupDecoder:
    """Decodes N videos in parallel, keyed by a shared global timeline.

    Each video has an associated time offset so that
    ``local_ts[i] = global_ts - offset[i]``.

    All blocking calls dispatch to a thread pool so the N decoders run
    concurrently.  Each individual ``VideoDecoder`` is only ever touched by
    one thread at a time — no per-decoder locking is needed.
    """

    def __init__(
        self,
        paths: list[Path | str],
        offsets: list[float],
        display_sizes: list[tuple[int, int] | None] | None = None,
    ) -> None:
        n = len(paths)
        if display_sizes is None:
            display_sizes = [None] * n
        self._offsets: list[float] = list(offsets)
        self._decoders: list[VideoDecoder] = [
            VideoDecoder(p, ds)
            for p, ds in zip(paths, display_sizes)
        ]
        self._executor = ThreadPoolExecutor(
            max_workers=n, thread_name_prefix="VideoGroup"
        )
        # Per-decoder bookkeeping for sequential-read optimisation.
        # _current_pts[i]  — local PTS of the last frame returned for decoder i
        # _last_frames[i]  — the last (frame, pts) tuple returned for decoder i
        self._current_pts: list[float] = [-1.0] * n
        self._last_frames: list[tuple[np.ndarray | None, float]] = [(None, -1.0)] * n
        logger.info(
            "VideoGroupDecoder: %d video(s) opened  hw=%s",
            n,
            [d.hw_active for d in self._decoders],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_frames(
        self, global_ts: float
    ) -> list[tuple[np.ndarray | None, float]]:
        """Blocking.  Force-seek all decoders to their local timestamps and return frames.

        ``local_ts[i] = global_ts - offset[i]``.
        Videos whose local timestamp is negative return ``(None, -1.0)``
        without touching their decoder (video not started yet at this position).

        Always uses ``decode_at`` (random seek) regardless of current position.
        Use ``advance_to_group`` for normal playback to take advantage of
        sequential reading.

        Returns a list of ``(rgb_frame, local_pts_s)`` — one entry per video,
        in the same order as the ``paths`` passed to ``__init__``.
        """
        def _decode_one(i: int) -> tuple[np.ndarray | None, float]:
            local_ts = global_ts - self._offsets[i]
            if local_ts < 0.0:
                self._last_frames[i] = (None, -1.0)
                self._current_pts[i] = -1.0
                return None, -1.0
            try:
                result = self._decoders[i].decode_at(local_ts)
            except Exception:
                logger.exception("VideoGroupDecoder.read_frames[%d] error", i)
                result = (None, -1.0)
            self._last_frames[i] = result
            self._current_pts[i] = result[1] if result[1] >= 0 else -1.0
            return result

        futures: list[Future] = [
            self._executor.submit(_decode_one, i)
            for i in range(len(self._decoders))
        ]
        return [f.result() for f in futures]

    def advance_to_group(
        self,
        global_ts: float,
        max_sequential_s: float = 2.0,
    ) -> tuple[list[tuple[np.ndarray | None, float]], bool]:
        """Blocking.  Advance all decoders to ``global_ts`` and return frames.

        For each decoder, the local target is ``local_ts = global_ts - offset[i]``.

        Strategy per decoder:
        - If ``local_ts < 0``: video not started yet → ``(None, -1.0)``.
        - If the stored frame is within one frame interval past the target
          (overshoot from the previous sequential advance), reuse it — this is
          the normal case for high-FPS sources (e.g. 120fps) where sequential
          advance always overshoots by up to one frame interval.
        - If ``local_ts`` is within ``max_sequential_s`` ahead of the last
          decoded PTS: advance by calling ``decode_next()`` in a loop (fast,
          no seek — takes advantage of sequential I/O).
        - Otherwise (uninitialized, behind by >1 frame, or too far ahead): fall
          back to ``decode_at`` (random seek).

        Returns ``(frames, had_seek)`` where ``had_seek`` is True if at least
        one decoder needed a random seek.
        """
        any_seek: list[bool] = [False]

        def _advance_one(i: int) -> tuple[np.ndarray | None, float]:
            local_target = global_ts - self._offsets[i]
            if local_target < 0.0:
                return None, -1.0

            current_pts = self._current_pts[i]
            fps = max(self._decoders[i]._fps, 1.0)
            delta = local_target - current_pts          # positive = target is ahead
            one_frame = 1.0 / fps
            half_frame = 0.5 / fps

            if current_pts < 0 or delta < -one_frame or delta > max_sequential_s:
                # Random seek required:
                # - uninitialized (first call after open/seek)
                # - stored frame is >1 frame past target (backward jump or big seek)
                # - target is >max_sequential_s ahead (forward jump)
                any_seek[0] = True
                try:
                    frame, pts = self._decoders[i].decode_at(local_target)
                except Exception:
                    logger.exception("VideoGroupDecoder.advance_to_group[%d] seek error", i)
                    return self._last_frames[i]
            elif delta <= half_frame:
                # Stored frame is already at or near the target.
                # Covers two cases:
                #   delta in (-one_frame, 0]  → sequential advance overshot by
                #     up to one frame (common for 60/120fps); reuse stored frame.
                #   delta in (0, half_frame]  → target barely ahead; not worth
                #     advancing one more frame.
                return self._last_frames[i]
            else:
                # Sequential advance — skip intermediate frames without reformatting.
                try:
                    frame, pts = self._decoders[i].advance_to(local_target)
                except Exception:
                    logger.exception(
                        "VideoGroupDecoder.advance_to_group[%d] sequential error", i
                    )
                    return self._last_frames[i]
                if frame is None:
                    return self._last_frames[i] if self._last_frames[i][0] is not None else (None, -1.0)

            self._last_frames[i] = (frame, pts)
            self._current_pts[i] = pts if pts >= 0 else current_pts
            return frame, pts

        futures: list[Future] = [
            self._executor.submit(_advance_one, i)
            for i in range(len(self._decoders))
        ]
        results = [f.result() for f in futures]
        return results, any_seek[0]

    def update_offsets(self, offsets: list[float]) -> None:
        self._offsets = list(offsets)

    @property
    def offsets(self) -> list[float]:
        return list(self._offsets)

    @property
    def current_global_ts(self) -> float:
        """Global timestamp of the last decoded frame of video 0, or -1.0 if unset."""
        pts = self._current_pts[0]
        return pts + self._offsets[0] if pts >= 0 else -1.0

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        for i, dec in enumerate(self._decoders):
            try:
                dec.close()
            except Exception:
                logger.exception("VideoGroupDecoder.close[%d] error", i)

    def __enter__(self) -> VideoGroupDecoder:
        return self

    def __exit__(self, *_) -> None:
        self.close()


def iter_decoded_frames(
        path: Path | str,
        n_frames: int,
        want_frame: Callable[[int], bool],
        prefetch: int = 8,
        should_stop: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, rgb)`` for wanted frames of one video, in order.

    A background thread decodes the video sequentially — skipping frames for
    which ``want_frame(idx)`` is False (cheap, no reformat) and reformatting the
    rest — and pushes them into a bounded queue. The caller iterates this
    generator and does its (GPU) work, so decode/reformat overlaps that work.

    The producer thread does no CUDA, so it overlaps GPU consumers freely. The
    queue is bounded by ``prefetch`` frames → the producer blocks when full,
    capping host memory. Producer exceptions are re-raised in the caller's
    thread; abandoning the generator early stops and joins the producer.
    """
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    stop_sentinel = object()
    err: list[Exception] = []
    abort = threading.Event()

    def _produce() -> None:
        try:
            with VideoDecoder(path) as dec:
                for idx in range(n_frames):
                    if abort.is_set() or (should_stop is not None and should_stop()):
                        break
                    if not want_frame(idx):
                        if not dec.skip_frame():
                            break
                        continue
                    rgb, _ = dec.decode_next()
                    if rgb is None:
                        break
                    q.put((idx, rgb))
        except Exception as exc:  # surface to consumer thread
            err.append(exc)
        finally:
            q.put(stop_sentinel)

    producer = threading.Thread(target=_produce, name="FramePrefetch", daemon=True)
    producer.start()

    stop_seen = False
    try:
        while True:
            item = q.get()
            if item is stop_sentinel:
                stop_seen = True
                break
            yield item
    finally:
        abort.set()
        if not stop_seen:
            # consumer abandoned early — drain so a blocked producer can exit
            while q.get() is not stop_sentinel:
                pass
        producer.join(timeout=5.0)
        if err:
            raise err[0]


def _open_nvdec(path: Path | str, gpu_id: int):
    """Open a PyNvVideoCodec GPU decoder (planar-RGB, device memory), or None if unavailable.

    Returns a decoder whose ``[i]`` yields a DLPack DecodedFrame (CHW uint8 RGB) resident on
    the GPU — NVDEC decode + NV12→RGB on-device, no host round-trip. None on missing package,
    no CUDA build, or open failure, so callers fall back to CPU decode.
    """
    try:
        import PyNvVideoCodec as nvc
    except Exception:
        return None
    try:
        return nvc.SimpleDecoder(str(path), gpu_id=gpu_id, use_device_memory=True,
                                 output_color_type=nvc.OutputColorType.RGBP)
    except Exception as exc:
        logger.warning("PyNvVideoCodec open failed for %s (%s); using CPU decode", path, exc)
        return None


def _open_threaded_nvdec(path: Path | str, gpu_id: int, buffer_size: int):
    """Open a PyNvVideoCodec ThreadedDecoder, or None if unavailable.

    Unlike SimpleDecoder's synchronous ``dec[i]`` (which syncs against the whole GPU and so collapses
    to ~30 fps when the engine/crop streams saturate the SMs), the ThreadedDecoder runs NVDEC on its
    OWN internal thread, decoding ``buffer_size`` frames AHEAD into device memory. That lets decode
    fill the GPU-idle gaps left by the pose engine instead of blocking on it — the streamed pose
    pipeline was decode-bound (decode-only 660 fps but only ~300 fps in-pipeline; measured pull_wait
    ≈ whole runtime). Frames are popped via ``get_batch_frames``.
    """
    try:
        import PyNvVideoCodec as nvc
    except Exception:
        return None
    try:
        return nvc.ThreadedDecoder(str(path), max(2, buffer_size), gpu_id=gpu_id,
                                   use_device_memory=True,
                                   output_color_type=nvc.OutputColorType.RGBP)
    except Exception as exc:
        logger.warning("PyNvVideoCodec ThreadedDecoder open failed for %s (%s); using SimpleDecoder",
                       path, exc)
        return None


def iter_frames_on_device(
        path: Path | str,
        n_frames: int,
        target_device,
        decode_device=None,
        want_frame: Callable[[int], bool] | None = None,
        prefetch: int = 8,
) -> Iterator[tuple[int, "object"]]:
    """Yield ``(frame_idx, image)`` as a CHW float32 tensor in [0, 1] on ``target_device``.

    GPU path (``decode_device`` is CUDA and PyNvVideoCodec is available): NVDEC decodes straight
    to a GPU RGB tensor — no swscale, no GPU→CPU→GPU round-trip. ``.float()`` allocates a fresh
    tensor, decoupling the result from the decoder's reused surface; ``.to(target_device)`` is a
    no-op when decode and target are the same GPU. CPU fallback: decode on host (the threaded
    ``iter_decoded_frames``) then upload to ``target_device``. ``decode_device`` defaults to
    ``target_device`` (GPU) or any available CUDA device.
    """
    import torch
    from kineo_sync.offline.image_utils import rgb_to_tensor

    target_device = torch.device(target_device)
    if decode_device is None:
        decode_device = (target_device if target_device.type == "cuda"
                         else (torch.device("cuda") if torch.cuda.is_available()
                               else torch.device("cpu")))
    else:
        decode_device = torch.device(decode_device)
    want = want_frame or (lambda i: True)

    if decode_device.type == "cuda":
        # ThreadedDecoder (decode-ahead) is the default — see _open_threaded_nvdec; it falls back to the
        # synchronous SimpleDecoder below if unavailable.
        convert = _frame_converter()
        td = _open_threaded_nvdec(path, decode_device.index or 0, buffer_size=max(8, prefetch))
        if td is not None:
            i = 0
            while i < n_frames:
                batch = td.get_batch_frames(8)             # pop up to 8 decoded-ahead frames
                if not batch:
                    break                                  # EOF
                for f in batch:
                    if i >= n_frames:
                        break
                    if want(i):
                        # convert() owns the surface (via .float()) before the buffer recycles it.
                        yield i, convert(torch.from_dlpack(f).to(target_device))
                    i += 1
            return

        dec = _open_nvdec(path, decode_device.index or 0)
        if dec is not None:
            n = min(n_frames, len(dec))
            for i in range(n):
                if not want(i):
                    continue
                # zero-copy DLPack view of the GPU surface; convert() copies it into owned memory
                # before the decoder reuses that surface for a later frame.
                yield i, convert(torch.from_dlpack(dec[i]).to(target_device))
            return

    # CPU fallback: host decode (threaded prefetch) + upload.
    for idx, rgb in iter_decoded_frames(path, n_frames, want_frame=want, prefetch=prefetch):
        yield idx, rgb_to_tensor(rgb, target_device)


def decode_frame_at(
        path: Path | str,
        timestamp_s: float,
        display_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    path = Path(path)
    timestamp_s = max(0.0, timestamp_s)
    t0 = time.perf_counter()
    logger.debug("decode_frame_at: %s @ %.4fs", path.name, timestamp_s)
    with VideoDecoder(path, display_size) as dec:
        frame, _ = dec.decode_at(timestamp_s)
    logger.debug("decode_frame_at: done in %.3fs", time.perf_counter() - t0)
    return frame
