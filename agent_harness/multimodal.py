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


def image_message(attachments: List[ImageAttachment]) -> Dict[str, Any]:
    """The user message that delivers images after a turn's tool results."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": "Image(s) from your view_image call(s): " + "; ".join(
        f"{a.path} ({a.info.width}x{a.info.height})" for a in attachments) + ". Report only what is actually legible; mark anything unclear as unreadable."}]
    parts += [a.part() for a in attachments]
    return {"role": "user", "content": parts}


def is_image_message(m: Dict[str, Any]) -> bool:
    return m.get("role") == "user" and isinstance(m.get("content"), list) and any(isinstance(x, dict) and x.get("type") == "image_url" for x in m["content"])


def prune_images(messages: List[Dict[str, Any]], keep: int = 2) -> int:
    """Replace the image parts of all but the newest ``keep`` image messages with a text note (indices and roles are unchanged). Returns how many were pruned."""
    idx = [i for i, m in enumerate(messages) if is_image_message(m)]
    n = 0
    for i in idx[:-keep] if keep else idx:
        old = messages[i]["content"]
        kept = [x for x in old if x.get("type") == "text"]
        kept.append({"type": "text", "text": "[the image was removed from the history to save context; call view_image on the file again to see it]"})
        messages[i] = {**messages[i], "content": kept}
        n += 1
    return n


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
