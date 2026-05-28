"""A small visual Polaroid scanner.

Run with:
    python polaroid_scanner_app.py

Required:
    Pillow

Optional:
    opencv-python, numpy

OpenCV is used only for automatic corner detection. The manual four-corner
workflow works with Pillow alone.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageOps, ImageStat, ImageTk


Point = Tuple[float, float]

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    PERSPECTIVE_METHOD = Image.Transform.PERSPECTIVE
except AttributeError:  # Pillow < 9
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_BICUBIC = Image.BICUBIC
    PERSPECTIVE_METHOD = Image.PERSPECTIVE


@dataclass
class DisplayTransform:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    width: int = 0
    height: int = 0


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polygon_area(points: Sequence[Point]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def order_points(points: Sequence[Point]) -> List[Point]:
    """Return points in top-left, top-right, bottom-right, bottom-left order."""
    pts = list(points)
    if len(pts) != 4:
        raise ValueError("Exactly four points are required.")

    top_left = min(pts, key=lambda p: p[0] + p[1])
    bottom_right = max(pts, key=lambda p: p[0] + p[1])
    top_right = max(pts, key=lambda p: p[0] - p[1])
    bottom_left = min(pts, key=lambda p: p[0] - p[1])
    return [top_left, top_right, bottom_right, bottom_left]


def solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    """Solve a small linear system with Gaussian elimination."""
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    size = len(rows)

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise ValueError("Corner points are degenerate; move them farther apart.")
        rows[column], rows[pivot] = rows[pivot], rows[column]

        pivot_value = rows[column][column]
        for item in range(column, size + 1):
            rows[column][item] /= pivot_value

        for row_index in range(size):
            if row_index == column:
                continue
            factor = rows[row_index][column]
            if factor == 0:
                continue
            for item in range(column, size + 1):
                rows[row_index][item] -= factor * rows[column][item]

    return [row[-1] for row in rows]


def find_perspective_coeffs(dst_points: Sequence[Point], src_points: Sequence[Point]) -> List[float]:
    """Return PIL coefficients mapping destination pixels to source pixels."""
    matrix: List[List[float]] = []
    vector: List[float] = []

    for (x, y), (u, v) in zip(dst_points, src_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        vector.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.append(v)

    return solve_linear_system(matrix, vector)


def warp_polaroid(image: Image.Image, corners: Sequence[Point]) -> Image.Image:
    ordered = order_points(corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    width_a = distance(bottom_right, bottom_left)
    width_b = distance(top_right, top_left)
    height_a = distance(top_right, bottom_right)
    height_b = distance(top_left, bottom_left)

    output_width = max(16, int(round(max(width_a, width_b))))
    output_height = max(16, int(round(max(height_a, height_b))))

    dst = [
        (0.0, 0.0),
        (float(output_width - 1), 0.0),
        (float(output_width - 1), float(output_height - 1)),
        (0.0, float(output_height - 1)),
    ]
    coeffs = find_perspective_coeffs(dst, ordered)
    return image.transform(
        (output_width, output_height),
        PERSPECTIVE_METHOD,
        coeffs,
        RESAMPLE_BICUBIC,
    )


def crop_inner_photo(image: Image.Image) -> Image.Image:
    """Estimate the inner photo area after perspective correction."""
    width, height = image.size
    left = int(width * 0.075)
    top = int(height * 0.075)
    right = int(width * 0.925)
    bottom = int(height * 0.785)

    if right - left < 16 or bottom - top < 16:
        return image
    return image.crop((left, top, right, bottom))


def mild_enhance(image: Image.Image) -> Image.Image:
    """Apply conservative scan-style enhancement."""
    rgb = image.convert("RGB")
    rgb = gray_world_balance(rgb)
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.08)
    return rgb


def gray_world_balance(image: Image.Image) -> Image.Image:
    stat = ImageStat.Stat(image)
    means = stat.mean[:3]
    if not all(value > 1 for value in means):
        return image

    target = sum(means) / 3.0
    channels = image.split()
    balanced = []

    for channel, mean in zip(channels[:3], means):
        scale = target / mean
        lut = [max(0, min(255, int(i * scale))) for i in range(256)]
        balanced.append(channel.point(lut))

    return Image.merge("RGB", balanced)


def reduce_glare(image: Image.Image, strength: int = 60) -> Image.Image:
    """Reduce specular glare with OpenCV inpainting and soft local blending."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "消除反光需要安装 OpenCV：python -m pip install opencv-python"
        ) from exc

    amount = max(0, min(100, int(strength))) / 100.0
    if amount <= 0:
        return image.convert("RGB")

    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    image_area = max(1, height * width)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    _hue, saturation, value = cv2.split(hsv)
    lightness, _a, _b = cv2.split(lab)

    sat_limit = min(145, int(58 + amount * 92))
    soft_value_limit = max(188, int(242 - amount * 54))
    soft_light_limit = max(188, int(238 - amount * 48))
    hard_value_limit = max(222, int(252 - amount * 20))
    hard_light_limit = max(220, int(250 - amount * 26))

    very_bright = (value >= 250) & (lightness >= 244)
    hard_mask = (
        ((value >= hard_value_limit) & (lightness >= hard_light_limit) & (saturation <= sat_limit))
        | very_bright
    ).astype("uint8") * 255
    soft_mask = (
        ((value >= soft_value_limit) & (lightness >= soft_light_limit) & (saturation <= sat_limit + 28))
        | hard_mask.astype(bool)
    ).astype("uint8") * 255

    hard_mask = filter_glare_mask(hard_mask, image_area, width, height, cv2, np)
    soft_mask = filter_glare_mask(soft_mask, image_area, width, height, cv2, np)

    kernel_size = max(3, int(3 + amount * 8))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if cv2.countNonZero(hard_mask) > 0:
        inpaint_mask = cv2.dilate(hard_mask, kernel, iterations=1)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        radius = max(3, int(3 + amount * 5))
        repaired_bgr = cv2.inpaint(bgr, inpaint_mask, radius, cv2.INPAINT_TELEA)
        repaired = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)
        soft_mask = cv2.bitwise_or(soft_mask, inpaint_mask)
    else:
        repaired = rgb

    if cv2.countNonZero(soft_mask) == 0:
        return Image.fromarray(repaired)

    soft_mask = cv2.dilate(soft_mask, kernel, iterations=1)
    alpha = cv2.GaussianBlur(soft_mask, (0, 0), sigmaX=kernel_size)
    alpha = (alpha.astype("float32") / 255.0)[:, :, None] * (0.28 + amount * 0.55)

    local_blur = cv2.GaussianBlur(repaired, (0, 0), sigmaX=5 + amount * 8)
    softened = repaired.astype("float32") * (1.0 - 0.10 * amount)
    softened = softened * (1.0 - 0.18 * amount) + local_blur.astype("float32") * (0.18 * amount)

    output = repaired.astype("float32") * (1.0 - alpha) + softened * alpha
    output = np.clip(output, 0, 255).astype("uint8")
    return Image.fromarray(output)


def filter_glare_mask(mask, image_area, image_width, image_height, cv2, np):
    """Keep compact highlight blobs and reject broad white border-like regions."""
    if cv2.countNonZero(mask) == 0:
        return mask

    filtered = np.zeros_like(mask)
    min_area = max(8, int(image_area * 0.00005))
    max_area = max(150, int(image_area * 0.16))
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    for label in range(1, labels_count):
        x, y, width, height, area = stats[label]
        if area < min_area or area > max_area:
            continue

        touches_edges = int(x <= 1) + int(y <= 1) + int(x + width >= image_width - 1) + int(y + height >= image_height - 1)
        looks_like_border_strip = (
            (width > image_width * 0.72 and height < image_height * 0.24)
            or (height > image_height * 0.72 and width < image_width * 0.24)
        )
        if looks_like_border_strip or (touches_edges >= 2 and area > image_area * 0.018):
            continue

        filtered[labels == label] = 255

    return filtered


def detect_polaroid_corners(image: Image.Image) -> Tuple[Optional[List[Point]], str]:
    """Detect a likely Polaroid outline with OpenCV when available."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None, "未安装 OpenCV，当前可使用手动四点扫描。"

    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    max_side = max(original_width, original_height)
    scale = min(1.0, 1200.0 / float(max_side))

    if scale < 1.0:
        scan_image = rgb.resize(
            (int(original_width * scale), int(original_height * scale)),
            RESAMPLE_LANCZOS,
        )
    else:
        scan_image = rgb

    arr = np.array(scan_image)
    height, width = arr.shape[:2]
    image_area = float(width * height)
    candidates: List[Tuple[float, List[Point]]] = []

    def score_candidate(points: Sequence[Point], source_bonus: float = 0.0) -> Optional[float]:
        try:
            ordered = order_points(points)
        except ValueError:
            return None

        quad_area = polygon_area(ordered)
        if quad_area < image_area * 0.025 or quad_area > image_area * 0.98:
            return None

        quad_width = max(distance(ordered[0], ordered[1]), distance(ordered[2], ordered[3]))
        quad_height = max(distance(ordered[1], ordered[2]), distance(ordered[3], ordered[0]))
        if quad_width < 30 or quad_height < 30:
            return None

        aspect = quad_width / quad_height
        if aspect < 0.45 or aspect > 2.2:
            return None

        rect_area = quad_width * quad_height
        rectangularity = min(1.0, quad_area / rect_area) if rect_area else 0.0
        if rectangularity < 0.45:
            return None

        aspect_penalty = min(abs(math.log(aspect / 0.78)), abs(math.log(aspect / 1.28)))
        area_score = quad_area / image_area
        return area_score * 3.5 + rectangularity * 1.4 - aspect_penalty * 0.35 + source_bonus

    def add_contours(contours: Iterable[object], source_bonus: float) -> None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.02 or area > image_area * 0.99:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            for epsilon in (0.012, 0.018, 0.026, 0.036, 0.05):
                approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    points = approx.reshape(4, 2).astype(float).tolist()
                    score = score_candidate(points, source_bonus)
                    if score is not None:
                        candidates.append((score, points))
                    break

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contours(contours, 0.0)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 135), (179, 75, 255))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contours(white_contours, 0.35)

    if not candidates and contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > image_area * 0.02:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            points = box.astype(float).tolist()
            score = score_candidate(points, -0.2)
            if score is not None:
                candidates.append((score, points))

    if not candidates:
        return None, "没有找到稳定的拍立得外框，可以手动拖动四个角点。"

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = order_points(candidates[0][1])
    if scale != 0:
        best = [(x / scale, y / scale) for x, y in best]

    return best, "已自动识别外框；如有偏差，可拖动角点微调。"


class PolaroidScannerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Polaroid Scanner")
        self.root.geometry("1180x760")
        self.root.minsize(920, 620)

        self.source_image: Optional[Image.Image] = None
        self.source_path: Optional[Path] = None
        self.result_image: Optional[Image.Image] = None
        self.corners: List[Point] = []
        self.selected_corner: Optional[int] = None

        self.source_tk: Optional[ImageTk.PhotoImage] = None
        self.result_tk: Optional[ImageTk.PhotoImage] = None
        self.source_transform = DisplayTransform()
        self.result_transform = DisplayTransform()

        self.output_mode = tk.StringVar(value="完整拍立得")
        self.enhance_var = tk.BooleanVar(value=True)
        self.glare_var = tk.BooleanVar(value=False)
        self.glare_strength_var = tk.IntVar(value=60)
        self.glare_strength_label_var = tk.StringVar(value="60")
        self.status_var = tk.StringVar(value="打开一张照片开始扫描。")

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Toolbar.TFrame", padding=(10, 8))
        style.configure("Status.TLabel", padding=(10, 6))

        toolbar = ttk.Frame(self.root, style="Toolbar.TFrame")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="打开图片", command=self.open_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="自动识别", command=self.auto_detect).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="重置角点", command=self.reset_corners).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="左转原图", command=lambda: self.rotate_source(90)).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="右转原图", command=lambda: self.rotate_source(-90)).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(toolbar, text="导出").pack(side=tk.LEFT, padx=(0, 6))
        mode_menu = ttk.Combobox(
            toolbar,
            textvariable=self.output_mode,
            values=("完整拍立得", "内部照片"),
            state="readonly",
            width=12,
        )
        mode_menu.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Checkbutton(toolbar, text="轻度增强", variable=self.enhance_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(toolbar, text="消除反光", variable=self.glare_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(toolbar, text="反光强度").pack(side=tk.LEFT, padx=(0, 4))
        glare_scale = ttk.Scale(
            toolbar,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.glare_strength_var,
            length=100,
            command=self.on_glare_strength_change,
        )
        glare_scale.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(toolbar, textvariable=self.glare_strength_label_var, width=3).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(toolbar, text="扫描预览", command=self.scan).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="保存结果", command=self.save_result).pack(side=tk.LEFT)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        source_frame = ttk.Frame(paned)
        result_frame = ttk.Frame(paned)
        paned.add(source_frame, weight=3)
        paned.add(result_frame, weight=2)

        ttk.Label(source_frame, text="原图：拖动四个角点调整扫描区域").pack(anchor=tk.W, pady=(0, 4))
        self.source_canvas = tk.Canvas(source_frame, bg="#141821", highlightthickness=0, cursor="crosshair")
        self.source_canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Label(result_frame, text="扫描结果").pack(anchor=tk.W, pady=(0, 4))
        self.result_canvas = tk.Canvas(result_frame, bg="#111827", highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.source_canvas.bind("<Configure>", lambda _event: self.redraw_source())
        self.result_canvas.bind("<Configure>", lambda _event: self.redraw_result())
        self.source_canvas.bind("<ButtonPress-1>", self.on_source_press)
        self.source_canvas.bind("<B1-Motion>", self.on_source_drag)
        self.source_canvas.bind("<ButtonRelease-1>", self.on_source_release)

        self.redraw_source()
        self.redraw_result()

    def on_glare_strength_change(self, value: str) -> None:
        self.glare_strength_var.set(int(float(value)))
        self.glare_strength_label_var.set(str(self.glare_strength_var.get()))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self.open_image())
        self.root.bind("<Control-s>", lambda _event: self.save_result())
        self.root.bind("<Return>", lambda _event: self.scan())

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择包含拍立得的照片",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ),
        )
        if not file_path:
            return

        try:
            image = Image.open(file_path)
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            messagebox.showerror("无法打开图片", str(exc))
            return

        self.source_path = Path(file_path)
        self.source_image = image
        self.result_image = None
        self.reset_corners(redraw=False)
        self.status_var.set(f"已打开：{self.source_path.name}。可自动识别，或直接拖动角点。")
        self.redraw_source()
        self.redraw_result()

    def reset_corners(self, redraw: bool = True) -> None:
        if not self.source_image:
            self.status_var.set("请先打开一张图片。")
            return

        width, height = self.source_image.size
        margin_x = width * 0.16
        margin_y = height * 0.14
        self.corners = [
            (margin_x, margin_y),
            (width - margin_x, margin_y),
            (width - margin_x, height - margin_y),
            (margin_x, height - margin_y),
        ]
        self.result_image = None
        self.status_var.set("已重置角点。拖动四个点包住拍立得外边框。")
        if redraw:
            self.redraw_source()
            self.redraw_result()

    def rotate_source(self, degrees: int) -> None:
        if not self.source_image:
            self.status_var.set("请先打开一张图片。")
            return
        self.source_image = self.source_image.rotate(degrees, expand=True)
        self.result_image = None
        self.reset_corners(redraw=False)
        self.status_var.set("已旋转原图，并重置角点。")
        self.redraw_source()
        self.redraw_result()

    def auto_detect(self) -> None:
        if not self.source_image:
            self.status_var.set("请先打开一张图片。")
            return

        self.status_var.set("正在自动识别拍立得外框...")
        self.root.update_idletasks()
        corners, message = detect_polaroid_corners(self.source_image)
        if corners:
            self.corners = corners
            self.result_image = None
        self.status_var.set(message)
        self.redraw_source()
        self.redraw_result()

    def scan(self) -> None:
        if not self.source_image:
            self.status_var.set("请先打开一张图片。")
            return
        if len(self.corners) != 4:
            self.status_var.set("需要四个角点才能扫描。")
            return

        try:
            result = warp_polaroid(self.source_image, self.corners)
            if self.glare_var.get():
                self.status_var.set("正在消除反光...")
                self.root.update_idletasks()
                result = reduce_glare(result, self.glare_strength_var.get())
            if self.output_mode.get() == "内部照片":
                result = crop_inner_photo(result)
            if self.enhance_var.get():
                result = mild_enhance(result)
        except ImportError as exc:
            messagebox.showinfo("需要安装 OpenCV", str(exc))
            self.status_var.set(str(exc))
            return
        except Exception as exc:
            messagebox.showerror("扫描失败", str(exc))
            return

        self.result_image = result
        self.status_var.set(f"扫描完成：{result.size[0]} x {result.size[1]} px。")
        self.redraw_result()

    def save_result(self) -> None:
        if not self.result_image:
            self.scan()
            if not self.result_image:
                return

        default_name = datetime.now().strftime("polaroid_scan_%Y%m%d_%H%M%S.jpg")
        output_path = filedialog.asksaveasfilename(
            title="保存扫描结果",
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=(("JPEG", "*.jpg"), ("PNG", "*.png")),
        )
        if not output_path:
            return

        path = Path(output_path)
        try:
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                self.result_image.convert("RGB").save(path, quality=95, subsampling=0)
            else:
                self.result_image.save(path)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return

        self.status_var.set(f"已保存：{path.name}")

    def redraw_source(self) -> None:
        canvas = self.source_canvas
        canvas.delete("all")
        if not self.source_image:
            self._draw_placeholder(canvas, "打开图片后在这里调整拍立得四个角")
            return

        displayed, transform = self._fit_image_to_canvas(self.source_image, canvas)
        self.source_transform = transform
        self.source_tk = ImageTk.PhotoImage(displayed)
        canvas.create_image(transform.offset_x, transform.offset_y, anchor=tk.NW, image=self.source_tk)

        if self.corners:
            display_points = [self.image_to_canvas(point) for point in self.corners]
            flat = [coord for point in display_points for coord in point]
            canvas.create_polygon(
                flat,
                outline="#38bdf8",
                fill="",
                width=2,
                dash=(8, 4),
            )
            for index, point in enumerate(display_points):
                radius = 8 if index != self.selected_corner else 10
                fill = "#f59e0b" if index != self.selected_corner else "#ef4444"
                canvas.create_oval(
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                    fill=fill,
                    outline="white",
                    width=2,
                )
                canvas.create_text(
                    point[0],
                    point[1] - 18,
                    text=str(index + 1),
                    fill="white",
                    font=("Segoe UI", 9, "bold"),
                )

    def redraw_result(self) -> None:
        canvas = self.result_canvas
        canvas.delete("all")
        if not self.result_image:
            self._draw_placeholder(canvas, "点击“扫描预览”后在这里查看结果")
            return

        displayed, transform = self._fit_image_to_canvas(self.result_image, canvas)
        self.result_transform = transform
        self.result_tk = ImageTk.PhotoImage(displayed)
        canvas.create_image(transform.offset_x, transform.offset_y, anchor=tk.NW, image=self.result_tk)

    def _fit_image_to_canvas(self, image: Image.Image, canvas: tk.Canvas) -> Tuple[Image.Image, DisplayTransform]:
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        padding = 24
        available_width = max(1, canvas_width - padding * 2)
        available_height = max(1, canvas_height - padding * 2)
        image_width, image_height = image.size
        scale = min(available_width / image_width, available_height / image_height)
        scale = min(scale, 1.0)
        display_width = max(1, int(image_width * scale))
        display_height = max(1, int(image_height * scale))
        offset_x = (canvas_width - display_width) / 2
        offset_y = (canvas_height - display_height) / 2
        displayed = image.resize((display_width, display_height), RESAMPLE_LANCZOS)
        transform = DisplayTransform(scale, offset_x, offset_y, display_width, display_height)
        return displayed, transform

    def _draw_placeholder(self, canvas: tk.Canvas, text: str) -> None:
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.create_text(
            width / 2,
            height / 2,
            text=text,
            fill="#cbd5e1",
            font=("Segoe UI", 12),
        )

    def image_to_canvas(self, point: Point) -> Point:
        transform = self.source_transform
        return (
            point[0] * transform.scale + transform.offset_x,
            point[1] * transform.scale + transform.offset_y,
        )

    def canvas_to_image(self, x: float, y: float) -> Point:
        transform = self.source_transform
        if transform.scale <= 0:
            return (0.0, 0.0)
        return (
            (x - transform.offset_x) / transform.scale,
            (y - transform.offset_y) / transform.scale,
        )

    def on_source_press(self, event: tk.Event) -> None:
        if not self.source_image or not self.corners:
            return

        display_points = [self.image_to_canvas(point) for point in self.corners]
        distances = [distance((event.x, event.y), point) for point in display_points]
        nearest = min(range(len(distances)), key=lambda index: distances[index])
        if distances[nearest] <= 22:
            self.selected_corner = nearest
            self.redraw_source()

    def on_source_drag(self, event: tk.Event) -> None:
        if self.selected_corner is None or not self.source_image:
            return

        image_x, image_y = self.canvas_to_image(event.x, event.y)
        width, height = self.source_image.size
        image_x = max(0.0, min(float(width - 1), image_x))
        image_y = max(0.0, min(float(height - 1), image_y))
        self.corners[self.selected_corner] = (image_x, image_y)
        self.result_image = None
        self.redraw_source()
        self.redraw_result()

    def on_source_release(self, _event: tk.Event) -> None:
        if self.selected_corner is not None:
            self.selected_corner = None
            self.status_var.set("角点已更新，点击“扫描预览”查看结果。")
            self.redraw_source()


def main() -> None:
    root = tk.Tk()
    app = PolaroidScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
