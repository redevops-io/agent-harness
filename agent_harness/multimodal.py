"""Image input for the agent loop: sniffing, size estimates, blank detection, OpenAI message parts, and a capability probe.

* Images enter the conversation only through a tool (``view_image``). A tool message cannot carry an image in the OpenAI format, so the loop appends one
  ``user`` message after the turn's tool results that holds the image part(s). Format verified against the pinned FP8 server (X0): content is a list of
  ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}`` and ``{"type": "text", ...}`` parts.
* Capability is never assumed from provider metadata: ``VisionCapability`` is set from explicit configuration, or measured by ``probe_vision`` with
  known images (a control image with random text, and a blank one).
* A blank or unreadable image is reported as such and is NOT attached: a model asked to read nothing will invent values (seen in X0), so the harness says
  "nothing to read" instead.
"""
from __future__ import annotations

import base64
import re
import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_EXTS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
PROBE_DIR = Path(__file__).resolve().parent / "data" / "vision_probe"


@dataclass(frozen=True)
class ImageInfo:
    mime: str
    width: int
    height: int
    size: int
    sha256: str


@dataclass
class ImageAttachment:
    path: str                       # path as the agent saw it
    data: bytes
    info: ImageInfo
    note: str = ""

    def part(self) -> Dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": f"data:{self.info.mime};base64," + base64.b64encode(self.data).decode()}}


def sniff_image(data: bytes) -> Optional[ImageInfo]:
    """mime and pixel size from the file header (PNG, JPEG, GIF, WebP) without any imaging library; None if not a recognised image."""
    w = h = 0
    mime = None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        mime, (w, h) = "image/png", struct.unpack(">II", data[16:24])
    elif data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        mime, (w, h) = "image/gif", struct.unpack("<HH", data[6:10])
    elif data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                break
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        mime = "image/webp"
        if data[12:16] == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            w, h = w & 0x3FFF, h & 0x3FFF
        elif data[12:16] == b"VP8L":
            b = struct.unpack("<I", data[21:25])[0]
            w, h = (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
        elif data[12:16] == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
    if not mime or w <= 0 or h <= 0:
        return None
    return ImageInfo(mime, int(w), int(h), len(data), hashlib.sha256(data).hexdigest())


def estimate_image_tokens(width: int, height: int) -> int:
    """About one token per 32x32 pixels (measured on the pinned server: a 600x400 image adds 230 prompt tokens)."""
    return max(16, (width * height) // 1024 - 4)


def blankness(data: bytes) -> str:
    """'blank' (one flat colour), 'faint' (almost no contrast), 'ok', or 'unknown' when no imaging library is available to decode it."""
    try:
        from PIL import Image, ImageStat
        import io
        im = Image.open(io.BytesIO(data)).convert("L")
        im.thumbnail((256, 256))
        lo, hi = im.getextrema()
        if hi - lo == 0:
            return "blank"
        sd = ImageStat.Stat(im).stddev[0]
        return "faint" if (hi - lo) < 24 or sd < 2.0 else "ok"
    except Exception:                                  # noqa: BLE001 - no PIL or an undecodable file
        return "unknown"


def load_image(path: Path, shown_path: str) -> Tuple[Optional[ImageAttachment], str]:
    """(attachment or None, message). None means: do not attach; the message says why (not an image, too large, blank, faint)."""
    p = Path(path)
    if not p.is_file():
        return None, f"no such file: {shown_path}"
    if p.stat().st_size > MAX_IMAGE_BYTES:
        return None, f"{shown_path} is {p.stat().st_size} bytes; images above {MAX_IMAGE_BYTES} bytes are not attached (resize it with python first)"
    data = p.read_bytes()
    info = sniff_image(data)
    if info is None:
        return None, f"{shown_path} is not a PNG, JPEG, GIF or WebP image"
    state = blankness(data)
    if state == "blank":
        return None, (f"{shown_path} ({info.width}x{info.height}) is a single flat colour: there is nothing to read in it. "
                      f"Report it as blank; do not infer or guess any values.")
    if state == "faint":
        return None, (f"{shown_path} ({info.width}x{info.height}) has almost no visible contrast, so it is not attached: nothing can be read reliably. "
                      f"Enhance it with python (contrast, threshold) and view the result, or report it as unreadable; do not guess values.")
    return ImageAttachment(shown_path, data, info), ""


def image_label(a: ImageAttachment) -> str:
    return f"[image: {a.path} ({a.info.width}x{a.info.height}) sha256:{a.info.sha256[:12]}]"


def image_message(attachments: List[ImageAttachment]) -> Dict[str, Any]:
    """The user message that delivers images after a turn's tool results. Each image is preceded by a text label (path, size, hash) that stays in the history
    if the image itself is later dropped."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": "Image(s) from your view_image call(s): " + "; ".join(
        f"{a.path} ({a.info.width}x{a.info.height})" for a in attachments) + ". Report only what is actually legible; mark anything unclear as unreadable."}]
    for a in attachments:
        parts += [{"type": "text", "text": image_label(a)}, a.part()]
    return {"role": "user", "content": parts}


def _is_image_part(x: Any) -> bool:
    return isinstance(x, dict) and x.get("type") == "image_url"


def is_image_message(m: Dict[str, Any]) -> bool:
    return m.get("role") == "user" and isinstance(m.get("content"), list) and any(_is_image_part(x) for x in m["content"])


def count_images(messages: List[Dict[str, Any]]) -> int:
    """Image content parts in the whole message list (what an endpoint's per-prompt image limit counts), not messages."""
    return sum(1 for m in messages if isinstance(m.get("content"), list) for x in m["content"] if _is_image_part(x))


class ImageCapError(RuntimeError):
    """The newest image message alone holds more images than the endpoint accepts; nothing older can be dropped to make room."""


def _drop_image(messages: List[Dict[str, Any]], i: int, j: int, reason: str) -> Dict[str, Any]:
    """Replace image part j of message i by a text note that keeps the label (path, size, hash). Returns {path/label, message index}."""
    parts = list(messages[i]["content"])
    label = parts[j - 1]["text"] if j > 0 and isinstance(parts[j - 1], dict) and parts[j - 1].get("type") == "text" and str(parts[j - 1].get("text", "")).startswith("[image: ") else "[image]"
    parts[j] = {"type": "text", "text": f"{label} was removed from the history ({reason}); it is not visible now. Use what you wrote down about it, or call view_image on the file again."}
    messages[i] = {**messages[i], "content": parts}
    return {"index": i, "label": label}


def _image_positions(messages: List[Dict[str, Any]]) -> List[tuple]:
    return [(i, j) for i, m in enumerate(messages) if isinstance(m.get("content"), list) for j, x in enumerate(m["content"]) if _is_image_part(x)]


def prune_images(messages: List[Dict[str, Any]], keep: int = 2) -> int:
    """Drop all but the newest ``keep`` images (parts, not messages) to save context. Indices and roles are unchanged. Returns how many were dropped."""
    pos = _image_positions(messages)
    drop = pos[:-keep] if keep else pos
    for i, j in drop:
        _drop_image(messages, i, j, "to save context")
    return len(drop)


def enforce_image_cap(messages: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Make the request satisfy an endpoint limit of ``cap`` images per prompt, whatever the token pressure: the oldest images are replaced by labelled text notes
    until at most ``cap`` remain. Returns what was dropped. The newest image message is never touched; if it alone exceeds the cap, raises ImageCapError so the
    caller can refuse the request locally with a clear reason instead of receiving an HTTP 400."""
    pos = _image_positions(messages)
    if len(pos) <= cap:
        return []
    newest = pos[-1][0]
    if sum(1 for i, _ in pos if i == newest) > cap:
        raise ImageCapError(f"the newest message holds {sum(1 for i, _ in pos if i == newest)} images but the endpoint accepts at most {cap} per prompt")
    dropped = []
    for i, j in pos[:len(pos) - cap]:
        if i == newest:
            raise ImageCapError(f"{len(pos)} images cannot be reduced to {cap} without dropping images of the newest message")
        dropped.append(_drop_image(messages, i, j, f"the endpoint accepts at most {cap} images per prompt"))
    return dropped


_LIMIT_RE = re.compile(r"at most (\d+) image", re.I)


def parse_image_limit(error: str) -> Optional[int]:
    """The per-prompt image limit an endpoint names in its error (vLLM: 'At most 5 image(s) may be provided in one prompt'), else None."""
    m = _LIMIT_RE.search(error or "")
    return int(m.group(1)) if m else None


def content_chars(content: Any) -> int:
    """Character-equivalent size of a message's content for the 3-chars-per-token estimate (images by their estimated tokens)."""
    if isinstance(content, str):
        return len(content)
    n = 0
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            raw = base64.b64decode(url.split(",", 1)[1][:200000] + "=" * 3, validate=False) if "," in url else b""
            info = sniff_image(raw)
            n += 3 * (estimate_image_tokens(info.width, info.height) if info else 1000)
        else:
            n += len(part.get("text") or "")
    return n


# ---- capability ----------------------------------------------------------------------------------------------------------------
@dataclass
class VisionCapability:
    """What the endpoint can do with images. ``source`` says how that is known: configuration | probe | assumed-none."""
    supported: bool = False
    source: str = "assumed-none"
    evidence: Dict[str, Any] = field(default_factory=dict)


def _fixture(name: str) -> bytes:
    return (PROBE_DIR / name).read_bytes()


CONTROL_EXPECTED = {"reference": "ZK-6305", "colour": "TEAL", "quantity": "61"}


def probe_vision(complete: Callable[[List[Dict[str, Any]]], str]) -> VisionCapability:
    """Measure vision with two known images. ``complete(messages) -> reply text``. Supported only if the control image's text is read back exactly
    AND the blank image's reply contains none of the control values."""
    control, blank = _fixture("control.png"), _fixture("blank.png")
    def ask(img: bytes) -> str:
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}}
        return complete([{"role": "user", "content": [part, {"type": "text", "text": "Read the text in the image. Reply with exactly three lines: Reference, Colour, Quantity."}]}])
    try:
        a, b = ask(control), ask(blank)
    except Exception as e:                              # noqa: BLE001 - a rejected image means no vision, with the reason recorded
        return VisionCapability(False, "probe", {"error": f"{type(e).__name__}: {e}"[:300]})
    reads = all(v.lower() in a.lower() for v in CONTROL_EXPECTED.values())
    leaks = any(v.lower() in b.lower() for v in CONTROL_EXPECTED.values())
    return VisionCapability(bool(reads and not leaks), "probe", {"control_reply": a[:200], "blank_reply": b[:200], "control_read_exactly": reads, "blank_leaked_control_values": leaks})
