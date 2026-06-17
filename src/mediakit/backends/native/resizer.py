
from PIL import Image

from mediakit.schemas.ops import ResizeMode


def _center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return img.crop((left, top, left + width, top + height))


def _saliency_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """Saliency-aware crop. Falls back to center-crop if OpenCV unavailable."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _center_crop(img, width, height)

    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()  # type: ignore[attr-defined]
    except AttributeError:
        return _center_crop(img, width, height)
    ok, sal_map = saliency.computeSaliency(bgr)
    if not ok:
        return _center_crop(img, width, height)

    sal_map_u8 = (sal_map * 255).astype(np.uint8)
    # bias toward faces if detectable
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
    )
    faces = face_cascade.detectMultiScale(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 1.1, 4)
    if len(faces) > 0:
        for x, y, w, h in faces:
            cv2.rectangle(sal_map_u8, (x, y), (x + w, y + h), 255, -1)

    # find centroid of salient region
    moments = cv2.moments(sal_map_u8)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = new_w // 2, new_h // 2

    left = max(0, min(cx - width // 2, new_w - width))
    top = max(0, min(cy - height // 2, new_h - height))
    return img.crop((left, top, left + width, top + height))


def _fit(img: Image.Image, width: int, height: int) -> Image.Image:
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    return img


def _pad(
    img: Image.Image, width: int, height: int, color: tuple[int, int, int]
) -> Image.Image:
    img = _fit(img.copy(), width, height)
    canvas = Image.new("RGB", (width, height), color)
    offset = ((width - img.width) // 2, (height - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def resize(
    img: Image.Image,
    width: int,
    height: int,
    mode: ResizeMode,
    pad_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    match mode:
        case ResizeMode.fit:
            return _fit(img.copy(), width, height)
        case ResizeMode.fill:
            return _center_crop(img, width, height)
        case ResizeMode.smart_crop:
            return _saliency_crop(img, width, height)
        case ResizeMode.pad:
            return _pad(img, width, height, pad_color)
