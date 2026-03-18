import sys
import os
import torch
import cv2
import numpy as np
import fitz  # PyMuPDF
import logging
from shapely.geometry import Polygon
from collections import Counter
from PIL import Image
from tqdm import tqdm

# --- 1. PATH & REGISTRY SETUP ---
DFINE_PATH = '/home/ubuntu/tumlinson_v2/tumlinson-electric/D-FINE'
if DFINE_PATH not in sys.path:
    sys.path.append(DFINE_PATH)

import src.zoo.dfine        # pyright: ignore
import src.nn.backbone      # pyright: ignore
import src.nn.postprocessor # pyright: ignore
from src.core import YAMLConfig  # pyright: ignore

# --- 2. CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GRAY_THRESHOLD     = 150
MIN_COMPONENT_AREA = 15

TILE_SIZE       = 640
OVERLAP_PERCENT = 0.20
STRIDE          = int(TILE_SIZE * (1 - OVERLAP_PERCENT))  # 512

# ── NMS ───────────────────────────────────────────────────────────────────────
# IoMin threshold: suppress if (intersection / min_area) > this value.
# This correctly handles tile-boundary duplicates where a partial box and a full
# box have low IoU but the smaller box is mostly inside the larger one.
IOMIN_THRES = 0.40

# Centre-distance fallback: suppress if centre gap < this * avg_max_dim.
# Kept as a cheap first check before the polygon operation.
CENTER_FRAC = 0.30

# ── Confidence ────────────────────────────────────────────────────────────────
CONF_THRES = 0.70

# CLASS_NAMES = {
#     1:  "balancing_valve",
#     2:  "ball_valve",
#     3:  "cap",
#     4:  "check_valve",
#     5:  "concentric_reducer",
#     6:  "elbow_down",
#     7:  "elbow_up",
#     8:  "floor_clean_out",
#     9:  "floor_drain",
#     10: "floor_sink",
#     11: "point_of_new_connection",
#     12: "roof_drain",
#     13: "shutoff_valve",
#     14: "solenoid_valve",
# }
CLASS_NAMES = {
    1:  "2x2_emergency_fixture",
    2:  "2x2_recessed_fixture",
    3:  "2x4_emergency_fixture",
    4:  "2x4_recessed_fixture",
    5:  "4_feet_emergency_light_surface_linear",
    6:  "4_foot_light_surface_fixture",
    7:  "8_feet_strip_pendant_light_fixture",
    8:  "air_terminal",
    9:  "aluminum_cable",
    10: "copper_cable",
    11: "data_outlet",
    12: "emergency_surface_mounted_strip_fixture",
    13: "equip_power_connection",
    14: "exit_sign",
    15: "junction_box",
    16: "light_emergency_battery",
    17: "light_emergency_fixture",
    18: "light_linear_recessed",
    19: "light_pendant",
    20: "light_recessed",
    21: "light_recessed_emergency",
    22: "light_recessed_square",
    23: "light_surface_linear",
    24: "light_wall_wash",
    25: "receptacle_duplex",
    26: "receptacle_duplex_ig",
    27: "receptacle_duplex_mounted",
    28: "receptacle_duplex_special",
    29: "receptacle_quad",
    30: "receptacle_quad_mounted",
    31: "single_convenience_receptacle",
    32: "special_purpose_outlet",
    33: "strip_emergency_light",
    34: "strip_light",
    35: "strip_pendant_emergency_light_fixture",
    36: "strip_pendant_light_fixture",
    37: "termination",
    38: "unknown",
    39: "wall_mounted_fixture"
}
#DFINE_MODEL_PATH  = "/home/ec2-user/Project/D-FINE/output/dfine_hgnetv2_s_custom_v2/best_stg1.pth"
DFINE_MODEL_PATH  = "/home/ubuntu/tumlinson_v2/tumlinson-electric/D-FINE/Infrence/best_stg2.pth"
DFINE_CONFIG_PATH = "/home/ubuntu/tumlinson_v2/tumlinson-electric/D-FINE/configs/dfine/custom/dfine_hgnetv2_s_custom.yml"

# --- 3. MODEL ---
class DFineModel:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = YAMLConfig(config_path, resume=model_path)
        self.model = self.cfg.model.to(self.device)
        self.postprocessor = self.cfg.postprocessor.to(self.device)
        if hasattr(self.postprocessor, "deploy"):
            self.postprocessor = self.postprocessor.deploy()

        checkpoint = torch.load(model_path, map_location=self.device)

        def _extract(obj):
            if not isinstance(obj, dict):
                return obj
            for key in ("static_graph", "model", "module"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
            return obj

        sd_src = "ema" if "ema" in checkpoint else ("model" if "model" in checkpoint else None)
        state_dict = _extract(checkpoint[sd_src] if sd_src else checkpoint)
        result = self.model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                f"Checkpoint mismatch — missing: {len(result.missing_keys)}, "
                f"unexpected: {len(result.unexpected_keys)}"
            )
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list:
        """
        Run inference on a 640x640 BGR tile.

        Score extraction is done carefully:
          - Move everything to CPU before converting to Python scalars.
          - Use .item() only on 0-d tensors; index explicitly otherwise.
          - Apply CONF_THRES as a strict >= comparison on a rounded float to
            eliminate any float16/bfloat16 representation edge cases.
        """
        orig_h, orig_w = image.shape[:2]
        img    = cv2.resize(image, (640, 640))
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        size   = torch.tensor([[orig_h, orig_w]], dtype=torch.long).to(self.device)

        out = self.model(tensor)
        labels_batch, boxes_batch, scores_batch = self.postprocessor(out, size)

        # Always work with the first (only) batch element, fully on CPU
        labels_np = labels_batch[0].cpu().numpy()   # shape: (N,)
        boxes_np  = boxes_batch[0].cpu().numpy()    # shape: (N, 4)
        scores_np = scores_batch[0].cpu().numpy()   # shape: (N,)

        # Log score range once per call to help diagnose threshold issues
        if len(scores_np) > 0:
            logger.debug(
                f"Raw scores — min: {scores_np.min():.4f}  max: {scores_np.max():.4f}  "
                f"above {CONF_THRES}: {(scores_np >= CONF_THRES).sum()}/{len(scores_np)}"
            )

        dets = []
        for i in range(len(scores_np)):
            # Round to 6 dp to eliminate any float16→float32 ghost decimals
            score = round(float(scores_np[i]), 6)
            if score < CONF_THRES:
                continue
            x1, y1, x2, y2 = boxes_np[i].tolist()
            dets.append({
                "class_id": int(labels_np[i]) + 1,
                "obb":      [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                "score":    score,
            })
        return dets


DFINE_MODEL = DFineModel(DFINE_MODEL_PATH, DFINE_CONFIG_PATH)


# --- 4. PREPROCESSING ---
def remove_gray_background(img: np.ndarray) -> np.ndarray:
    bgr  = img.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < MIN_COMPONENT_AREA:
            cv2.drawContours(mask, [c], -1, 0, -1)

    out = np.full_like(bgr, 255)
    out[mask == 255] = bgr[mask == 255]
    return out


# --- 5. TILING ---
def tile_image(img: np.ndarray, tile_size: int = TILE_SIZE, stride: int = STRIDE):
    h, w = img.shape[:2]
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not xs: xs = [0]
    if not ys: ys = [0]
    if xs[-1] + tile_size < w: xs.append(w - tile_size)
    if ys[-1] + tile_size < h: ys.append(h - tile_size)

    tiles, positions = [], []
    for y in ys:
        for x in xs:
            tile = img[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
                positions.append((x, y))
    return tiles, positions


# --- 6. NMS (IoMin-based) ---
def _obb_stats(obb: list) -> tuple:
    """Return (cx, cy, max_dim, shapely_poly) for an OBB."""
    xs  = [p[0] for p in obb]
    ys  = [p[1] for p in obb]
    w   = max(xs) - min(xs)
    h   = max(ys) - min(ys)
    cx  = min(xs) + w / 2
    cy  = min(ys) + h / 2
    return cx, cy, max(w, h), Polygon(obb)


def obb_nms(detections: list, iomin_thres: float = IOMIN_THRES) -> list:
    """
    Greedy NMS using IoMin (intersection / min_area) instead of IoU.

    WHY IoMin instead of IoU:
      At tile boundaries, the *same* symbol is often detected twice:
        - Tile A: full box covering the whole symbol  (area = A_full)
        - Tile B: partial box covering only the part  (area = A_part, much smaller)
      Their IoU = inter / (A_full + A_part - inter) can be as low as 0.10–0.20
      even when inter == A_part (i.e., the smaller box is entirely inside the
      larger one). IoMin = inter / min(A_full, A_part) = 1.0 in that case,
      so it correctly identifies them as duplicates.

    Suppression logic (same class required, then either condition triggers):
      A. Centre distance  < CENTER_FRAC * avg_max_dim   (cheap, runs first)
      B. IoMin            > iomin_thres                 (runs only when A misses)

    Post-filter: hard confidence floor applied after greedy selection to catch
    any edge case that slipped through.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        best_cx, best_cy, best_max, best_poly = _obb_stats(best["obb"])
        best_area = best_poly.area
        to_remove = []

        for i, det in enumerate(detections):
            if det["class_id"] != best["class_id"]:
                continue

            det_cx, det_cy, det_max, det_poly = _obb_stats(det["obb"])

            # --- A: centre proximity (no polygon op needed) ---
            avg_max     = (best_max + det_max) / 2
            centre_dist = np.hypot(det_cx - best_cx, det_cy - best_cy)
            if centre_dist < CENTER_FRAC * avg_max:
                to_remove.append(i)
                continue

            # --- B: IoMin ---
            inter    = best_poly.intersection(det_poly).area
            min_area = min(best_area, det_poly.area)
            iomin    = inter / min_area if min_area > 0 else 0.0
            if iomin > iomin_thres:
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            del detections[i]

    # Hard post-NMS confidence floor — belt-and-suspenders guarantee
    filtered = [d for d in keep if d["score"] >= CONF_THRES]

    suppressed = len(keep) - len(filtered)
    if suppressed:
        logger.info(f"Post-NMS confidence filter removed {suppressed} sub-threshold detections")

    return filtered


# --- 7. VISUALISATION ---
def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    draw_img     = img.copy()
    placed_rects = []
    font         = cv2.FONT_HERSHEY_SIMPLEX
    font_scale   = 0.4
    thickness    = 1

    detections = sorted(
        detections,
        key=lambda d: (min(p[1] for p in d["obb"]), min(p[0] for p in d["obb"]))
    )

    for det in detections:
        pts = np.array(det["obb"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(draw_img, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

        label     = f"{CLASS_NAMES.get(det['class_id'], 'Unknown')} {det['score']:.2f}"
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        xmin = min(p[0] for p in det["obb"])
        ymin = min(p[1] for p in det["obb"])
        xmax = max(p[0] for p in det["obb"])
        ymax = max(p[1] for p in det["obb"])
        box_cx = (xmin + xmax) / 2
        box_cy = (ymin + ymax) / 2

        offset     = 20
        candidates = [
            (int(xmin),                         int(ymin - offset)),
            (int(xmin),                         int(ymax + text_size[1] + offset)),
            (int(xmin - text_size[0] - offset), int(ymin)),
            (int(xmax + offset),                int(ymin)),
        ]

        chosen_pos = label_rect = None
        for lx, ly in candidates:
            r  = (lx, ly - text_size[1], lx + text_size[0], ly)
            ok = all(
                not (max(r[0] - 5, pr[0]) < min(r[2] + 5, pr[2]) and
                     max(r[1] - 5, pr[1]) < min(r[3] + 5, pr[3]))
                for pr in placed_rects
            )
            if ok:
                chosen_pos = (lx, ly)
                label_rect = r
                break

        if chosen_pos is None:
            lx, ly     = int(xmax + offset + 20), int(ymin)
            chosen_pos = (lx, ly)
            label_rect = (lx, ly - text_size[1], lx + text_size[0], ly)

        cv2.rectangle(draw_img,
                      (label_rect[0] - 2, label_rect[1] - 2),
                      (label_rect[2] + 2, label_rect[3] + 2),
                      (0, 0, 0), -1)
        cv2.putText(draw_img, label, chosen_pos, font, font_scale, (0, 255, 0), thickness)

        lbl_cx = (label_rect[0] + label_rect[2]) / 2
        lbl_cy = (label_rect[1] + label_rect[3]) / 2
        cv2.line(draw_img,
                 (int(lbl_cx), int(lbl_cy)),
                 (int(box_cx), int(box_cy)),
                 (255, 255, 0), 1)
        placed_rects.append(label_rect)

    return draw_img


# --- 8. PAGE PIPELINE ---
def process_page_for_symbols(pdf_path: str, page_num: int, scale: float = 4.0):
    doc  = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix  = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img  = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()

    logger.info(f"Page {page_num}: rendered {img.shape[1]}x{img.shape[0]} px")

    cleaned      = remove_gray_background(img)
    tiles, poses = tile_image(cleaned)
    logger.info(f"Page {page_num}: {len(tiles)} tiles")

    all_dets = []
    for tile, (tx, ty) in zip(tiles, poses):
        for det in DFINE_MODEL.detect(tile):
            det["obb"] = [(px + tx, py + ty) for px, py in det["obb"]]
            all_dets.append(det)

    logger.info(f"Page {page_num}: {len(all_dets)} raw detections → NMS")
    unique_dets = obb_nms(all_dets)
    logger.info(f"Page {page_num}: {len(unique_dets)} final detections (score >= {CONF_THRES})")

    counts    = Counter(d["class_id"] for d in unique_dets)
    drawn_img = draw_detections(cleaned, unique_dets)
    return Image.fromarray(cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)), counts, unique_dets


# --- 9. BATCH ENTRY POINT ---
def execute_symbol_detection(pdf_path: str, page_nums: list):
    per_page = {}
    total    = Counter()
    for page_num in tqdm(page_nums, desc="Detecting Symbols"):
        drawn, counts, dets = process_page_for_symbols(pdf_path, page_num)
        per_page[page_num] = {
            "drawn_image": drawn,
            "counts":      {CLASS_NAMES.get(k, f"Class_{k}"): v for k, v in counts.items()},
            "detections":  dets,
        }
        total.update(counts)
    total_named = {CLASS_NAMES.get(k, f"Class_{k}"): v for k, v in total.items()}
    return per_page, total_named


# --- 10. STANDALONE TEST ---
if __name__ == "__main__":
    # Enable debug logging to see score distributions
    logging.getLogger().setLevel(logging.DEBUG)

    pdf            = "/home/ubuntu/tumlinson_v2/tumlinson-electric/test/E2.05 95_ CD ELECTRICAL LIGHTING PLAN -LEVEL 2 -AREA 4.pdf"
    selected_pages = [1]
    results, summary = execute_symbol_detection(pdf, selected_pages)
    if selected_pages:
        results[selected_pages[0]]["drawn_image"].save("detection_test.png")
        print(f"Saved detection_test.png  |  Summary: {summary}")

        # Sanity check — assert no sub-threshold detections survived
        for det in results[selected_pages[0]]["detections"]:
            assert det["score"] >= CONF_THRES, \
                f"BUG: detection with score {det['score']:.6f} survived CONF_THRES={CONF_THRES}"
        print(f"✅ All {len(results[selected_pages[0]]['detections'])} detections confirmed >= {CONF_THRES}")