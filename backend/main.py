# main.py
"""
Automated OMR Evaluation - single-file pipeline (heuristic-first, optional classifier)
Requirements (install with pip):
    pip install opencv-python numpy pillow tqdm pymupdf torchvision torch scikit-learn
Note: torch/torchvision only required if you plan to use the optional classifier.
Usage:
    python main.py --input_dir ./uploads --out_dir ./out --answer_key answer_key.json
Outputs:
    ./out/<image_basename>_overlay.png
    ./out/<image_basename>_result.json
If you supply --model_path classifier.pth the pipeline will use the classifier for ambiguous bubbles.
"""

import os
import cv2
import sys
import json
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
try:
    import pytesseract
    _tess_cmd = os.environ.get('TESSERACT_CMD')
    if _tess_cmd and os.path.exists(_tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except Exception:
    pytesseract = None

# ---------------------------
# Excel Answer Key Loader (sheet or column mode)
# ---------------------------
def _normalize_set_name(name: str) -> str:
    c = name.replace('-', ' ').replace('_',' ').strip().lower()
    if c in {'set a','a'}:
        return 'Set_A'
    if c in {'set b','b'}:
        return 'Set_B'
    return name

def load_answer_keys_from_excel(path: str) -> Optional[Dict[str, Dict[str, List[str]]]]:
    """Return mapping of set/column -> {qno: [answers]} plus optional GLOBAL.
    Supports workbook with sheets named like Set-A / Set-B, or single-sheet multi-column.
    """
    def _parse_series(series, global_key):
        parsed: Dict[str,List[str]] = {}
        for raw in series.dropna():
            token = str(raw).strip()
            if not token:
                continue
            i = 0
            while i < len(token) and token[i].isdigit():
                i += 1
            if i == 0:
                continue
            qno = token[:i]
            remainder = token[i:].lstrip(' -.')
            if not remainder:
                continue
            answers = [a.strip().lower() for a in remainder.replace(' ', '').split(',') if a.strip()]
            if not answers:
                continue
            parsed[qno] = answers
            if qno not in global_key:
                global_key[qno] = answers
        return parsed
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] Could not open Excel answer key: {e}")
        return None
    sheet_names_lower = [s.lower() for s in xls.sheet_names]
    sheet_mode = any('set' in s for s in sheet_names_lower)
    answer_keys: Dict[str, Dict[str, List[str]]] = {}
    global_key: Dict[str, List[str]] = {}
    if sheet_mode:
        print('[INFO] (Excel) sheet-based answer key mode.')
        for sheet in xls.sheet_names:
            norm = _normalize_set_name(sheet)
            try:
                df_sheet = xls.parse(sheet, header=None)
            except Exception as e:
                print(f"[WARN] Failed reading sheet {sheet}: {e}")
                continue
            flat_vals = [v for v in df_sheet.fillna('').values.flatten() if str(v).strip()]
            series = pd.Series(flat_vals)
            parsed = _parse_series(series, global_key)
            if parsed:
                answer_keys[norm] = parsed
        if not answer_keys:
            print('[WARN] Sheet mode produced no keys; falling back to first sheet columns.')
            sheet_mode = False
    if not sheet_mode:
        try:
            df = xls.parse(xls.sheet_names[0], header=0)
        except Exception as e:
            print(f"[WARN] Column mode failed: {e}")
            return None
        for col in df.columns:
            series = df[col]
            parsed = _parse_series(series, global_key)
            if parsed:
                answer_keys[col] = parsed
    if global_key:
        answer_keys['GLOBAL'] = global_key
    if not answer_keys:
        print('[WARN] No valid QA patterns found in Excel.')
        return None
    print(f"[INFO] Loaded Excel answer key sets: {list(answer_keys.keys())}")
    return answer_keys

def extract_set_code(image) -> Optional[str]:
    """OCR header region to detect 'Set' letter."""
    if pytesseract is None:
        return None
    try:
        h, w = image.shape[:2]
        header = image[0:int(h*0.25), :]
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        if not text:
            return None
        tl = text.lower().replace('\n',' ')
        import re
        m = re.search(r'set\s*(no\.?:|number:|:)??\s*([a-z])', tl)
        if m:
            return m.group(2).upper()
        m2 = re.search(r'set[^a-z0-9]{1,5}([a-z])\b', tl)
        if m2:
            return m2.group(1).upper()
    except Exception:
        return None
    return None

# Optional torch import (only needed if --model_path provided)
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------------------------
# Dataset Auto-Detection & Paths
# ---------------------------
DATASET_CANDIDATES = [
    Path(__file__).resolve().parent.parent / 'dataset' / 'data',
    Path(__file__).resolve().parent.parent / 'dataset' / 'Theme 1 - Sample Data',
    Path(__file__).resolve().parent.parent / 'dataset'
]

def detect_dataset_root() -> Path:
    for cand in DATASET_CANDIDATES:
        if cand.exists():
            # simple heuristic: contains at least one image or subdir
            try:
                for p in cand.iterdir():
                    return cand
            except Exception:
                pass
    return DATASET_CANDIDATES[-1]

DATASET_ROOT = detect_dataset_root()

def auto_key_file(root: Path) -> Optional[Path]:
    patterns = ['answer_key.json', 'key.json']
    for pat in patterns:
        cand = root / pat
        if cand.exists():
            return cand
    # generic search
    for f in root.glob('*.json'):
        if 'key' in f.name.lower():
            return f
    return None

# ---------------------------
# Utility & preprocessing
# ---------------------------
def read_image(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.pdf']:
        # lightweight PDF page -> image using PyMuPDF if available
        try:
            import fitz
            doc = fitz.open(path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            mode = "RGB" if pix.n < 4 else "RGBA"
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            return arr
        except Exception:
            raise RuntimeError("PyMuPDF required to read PDFs. Install 'pymupdf'.")
    else:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        return img

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def estimate_gamma(gray: np.ndarray, low=0.2, high=5.0):
    # crude estimate: mean brightness -> target ~0.5
    m = np.mean(gray) / 255.0
    if m <= 0: return 1.0
    gamma = math.log(0.5) / math.log(m)
    gamma = max(low, min(high, gamma))
    return gamma

def adjust_gamma(img: np.ndarray, gamma: float):
    inv = 1.0 / gamma
    table = np.array([((i/255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess_for_detection(img: np.ndarray) -> np.ndarray:
    """Return a processed grayscale image suitable for contour/grid detection"""
    gray = to_gray(img)
    clahe = apply_clahe(gray)
    gamma = estimate_gamma(clahe)
    adjusted = adjust_gamma(clahe, gamma)
    # slight blur to reduce small noise but keep edges
    blurred = cv2.GaussianBlur(adjusted, (3,3), 0)
    return blurred

# ---------------------------
# Sheet detection & warp
# ---------------------------
def find_largest_quad_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    edged = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:12]:  # check top contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            return approx
    return None

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4,2).astype("float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(img: np.ndarray, pts: np.ndarray, output_size: Tuple[int,int]=(2480,3508)) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    # choose either computed size or fixed output_size scaling
    dst = np.array([[0,0],[output_size[0]-1,0],[output_size[0]-1,output_size[1]-1],[0,output_size[1]-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, output_size)
    return warped

def warp_sheet(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Attempt to find sheet and warp. Returns warped image and success flag.
       If detection fails returns original resized image and False."""
    processed = preprocess_for_detection(img)
    quad = find_largest_quad_contour(processed)
    h, w = img.shape[:2]
    # fallback: assume image is sheet but scale to canonical size
    if quad is None:
        # try thresholding + morphological close to strengthen existing boundary and try again
        _, thr = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts[:10]:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                if len(approx) == 4 and cv2.contourArea(approx) > 8000:
                    quad = approx
                    break
    if quad is not None:
        try:
            warped = four_point_transform(img, quad, output_size=(1240,1754))  # scale down for speed
            return warped, True
        except Exception:
            pass
    # fallback: just resize while keeping aspect as sheet
    scale = 1240.0 / w
    new_h = int(h * scale)
    warped = cv2.resize(img, (1240, new_h))
    return warped, False

# ---------------------------
# Bubble detection helpers
# ---------------------------
def adaptive_binary(img_gray: np.ndarray) -> np.ndarray:
    # Use adaptive thresholding on local window to handle uneven lighting
    thr = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, blockSize=31, C=12)
    # morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def detect_bubble_contours(bin_img: np.ndarray, min_area=80, max_area=5000) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        # filter by circularity and area
        if circularity > 0.35:
            bubbles.append(c)
    return bubbles

def contour_center_radius(c: np.ndarray) -> Tuple[int,int,int]:
    (x,y), r = cv2.minEnclosingCircle(c)
    return int(x), int(y), int(r)

def sort_bubbles_grid(centers: List[Tuple[int,int,int]], approx_rows: Optional[int]=None) -> List[List[Tuple[int,int,int]]]:
    """
    Group centers into rows by y coordinate, then sort columns by x.
    Returns list-of-rows, each row is list of centers sorted by x.
    """
    if not centers:
        return []
    # convert to numpy array of centers
    pts = np.array([[c[0], c[1]] for c in centers])
    ys = pts[:,1]
    # cluster by y using simple greedy grouping with tolerance based on median radius
    radii = np.array([c[2] for c in centers])
    med_r = max(4, int(np.median(radii)))
    tol = med_r * 2.5  # vertical tolerance to group into same row
    sorted_idx = np.argsort(ys)
    rows = []
    current_row = [centers[sorted_idx[0]]]
    last_y = ys[sorted_idx[0]]
    for idx in sorted_idx[1:]:
        y = ys[idx]
        if abs(y - last_y) <= tol:
            current_row.append(centers[idx])
            last_y = (last_y * (len(current_row)-1) + y) / len(current_row)
        else:
            # finish current row
            # sort current row by x
            current_row = sorted(current_row, key=lambda c: c[0])
            rows.append(current_row)
            current_row = [centers[idx]]
            last_y = y
    if current_row:
        current_row = sorted(current_row, key=lambda c: c[0])
        rows.append(current_row)
    # Optionally, if approx_rows provided and rows are mis-split, you could merge/split further. Keep simple for now.
    return rows

# ---------------------------
# Patch extraction and heuristics
# ---------------------------
def crop_patch(img: np.ndarray, center: Tuple[int,int,int], size: int=48) -> np.ndarray:
    x, y, r = center
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    patch = img[y1:y2, x1:x2]
    # If patch is smaller than required, pad with white
    ph, pw = patch.shape[:2]
    if ph == 0 or pw == 0:
        return np.ones((size, size, 3), dtype=np.uint8) * 255
    if ph != size or pw != size:
        pad_img = np.ones((size, size, 3), dtype=np.uint8) * 255
        pad_img[:ph, :pw] = patch
        return pad_img
    return patch

def bubble_fill_ratio(patch: np.ndarray) -> float:
    # compute proportion of dark pixels in the patch
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Otsu threshold to make it robust
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thr == 0 are dark pixels here (if we use BINARY)
    # but ensure direction: we want filled area darker than background (common)
    # Count dark pixels
    dark = np.sum(thr == 0)
    total = thr.size
    return dark / total

# ---------------------------
# Optional classifier wrapper
# ---------------------------
class SmallBubbleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)  # 0: unmarked, 1: marked, 2: erased/ambiguous
        )
    def forward(self, x):
        return self.net(x)

def load_classifier(model_path: str, device='cpu'):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch not available. Install torch to use classifier.")
    model = SmallBubbleNet()
    ck = torch.load(model_path, map_location=device)
    # if the saved object is state_dict or full model handle both
    if isinstance(ck, dict) and 'state_dict' in ck:
        model.load_state_dict(ck['state_dict'])
    elif isinstance(ck, dict):
        # assume it's state_dict
        try:
            model.load_state_dict(ck)
        except Exception:
            # sometimes saved as entire model
            model = ck
    else:
        model = ck
    model.to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    return model, preprocess, device

def classifier_predict(model, preprocess, device, patch: np.ndarray) -> Tuple[int, float]:
    """
    Returns predicted class and confidence (softmax top1)
    classes: 0-unmarked,1-marked,2-erased
    """
    img = preprocess(patch)  # C x H x W, float [0,1]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
    return cls, conf

# ---------------------------
# Scoring helpers
# ---------------------------
def load_answer_key(path: str) -> Dict:
    """
    The expected format is JSON:
    {
      "questions": [
         {"qno": 1, "subject": "Math", "correct": "A"},
         ...
      ],
      "options_per_question": 4
    }
    or a dict mapping "1":"A", "2":"B", ...
    """
    with open(path, 'r') as f:
        data = json.load(f)
    # unify format
    if isinstance(data, dict) and "questions" in data:
        mapping = {}
        for q in data['questions']:
            mapping[str(q['qno'])] = q['correct']
        return mapping
    # else assume simple mapping
    return {str(k): v for k, v in data.items()}

def _select_excel_key(answer_sets: Dict[str, Dict[str, List[str]]], logical: str) -> Optional[str]:
    variants = [logical, logical.replace('_',' '), logical.upper(), logical.replace('_',' ').title()]
    for v in variants:
        if v in answer_sets:
            return v
    if 'GLOBAL' in answer_sets:
        return 'GLOBAL'
    return None

def compute_score(detected_answers: Dict[str,str], answer_key: Dict[str,str]) -> Dict:
    """
    detected_answers: {"1":"A", "2":null, "3":"B", ...}
    answer_key: {"1":"A", "2":"B", ...}
    returns score summary
    """
    total = 0
    correct = 0
    incorrect = 0
    unanswered = 0
    per_question = {}
    for qno, ans in answer_key.items():
        total += 1
        detected = detected_answers.get(str(qno))
        if detected is None:
            unanswered += 1
            per_question[str(qno)] = {"detected": None, "correct": ans, "status": "unanswered"}
        elif detected == ans:
            correct += 1
            per_question[str(qno)] = {"detected": detected, "correct": ans, "status": "correct"}
        else:
            incorrect += 1
            per_question[str(qno)] = {"detected": detected, "correct": ans, "status": "incorrect"}
    score = {"total": total, "correct": correct, "incorrect": incorrect, "unanswered": unanswered}
    return {"score": score, "per_question": per_question}

# ---------------------------
# Main pipeline: process one file
# ---------------------------
def process_single_image(path: str,
                         out_dir: str,
                         model_tuple=None,
                         answer_key: Optional[Dict[str,str]]=None,
                         debug: bool=False):
    basename = os.path.splitext(os.path.basename(path))[0]
    img = read_image(path)
    orig = img.copy()
    warped, detected_sheet = warp_sheet(img)
    # convert warped to rgb for consistent patch behavior
    warped_rgb = warped.copy() if warped.ndim == 3 else cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    gray = to_gray(warped_rgb)
    bin_img = adaptive_binary(gray)
    # detect bubble contours
    bubbles = detect_bubble_contours(bin_img)
    centers = [contour_center_radius(c) for c in bubbles]
    # if no bubble detected, try alternate parameters
    if len(centers) < 4:
        # try larger kernel thresholds
        _, thr2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr2 = cv2.bitwise_not(thr2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        thr2 = cv2.morphologyEx(thr2, cv2.MORPH_OPEN, kernel, iterations=1)
        bubbles2 = detect_bubble_contours(thr2, min_area=40, max_area=8000)
        centers = [contour_center_radius(c) for c in bubbles2]
    # cluster into rows
    rows = sort_bubbles_grid(centers)
    # if rows are too few, try relaxing tolerance by re-calling bubble detection with different area
    detected_answers = {}  # placeholder mapping qno -> option letter (A/B/C/D)
    bubble_results = []  # save per bubble info for audit
    # simple assumption: each row corresponds to a single question and columns to choices
    # So for a robust real pipeline you should map rows to questions using known layout
    q_counter = 1
    for row in rows:
        # ensure row has at least 2 columns; if too many skip if inconsistent
        ncols = len(row)
        if ncols < 2:
            # maybe merge with neighbors later; skip for now
            continue
        # sort by x
        row_sorted = sorted(row, key=lambda c: c[0])
        # treat each column as option A,B,C...
        option_letters = [chr(ord('A') + i) for i in range(ncols)]
        # compute fill ratios for each
        fill_list = []
        for c in row_sorted:
            patch = crop_patch(warped_rgb, c, size=48)
            fr = bubble_fill_ratio(patch)
            fill_list.append((c, fr, patch))
        # heuristic thresholds
        # high -> definitely marked; low -> definitely empty; between -> ambiguous
        high_th = 0.22  # these thresholds tuned experimentally; may need adjustment
        low_th = 0.06
        chosen = []
        ambigs = []
        for idx, (c, fr, patch) in enumerate(fill_list):
            status = None
            if fr >= high_th:
                chosen.append((idx, c, fr))
                status = "marked"
            elif fr <= low_th:
                status = "empty"
            else:
                status = "ambiguous"
                ambigs.append((idx, c, fr, patch))
            bubble_results.append({
                "q_no": None,  # fill later
                "option": option_letters[idx],
                "center": (int(c[0]), int(c[1])),
                "radius": int(c[2]),
                "fill_ratio": float(fr),
                "status": status
            })
        # if exactly one chosen -> that is the answer
        detected_option = None
        ambiguous_flag = False
        if len(chosen) == 1:
            detected_option = option_letters[chosen[0][0]]
        elif len(chosen) > 1:
            # multiple selections -> ambiguous
            ambiguous_flag = True
            # pick the highest fill ratio but mark as multiple
            chosen_sorted = sorted(chosen, key=lambda x: x[2], reverse=True)
            detected_option = option_letters[chosen_sorted[0][0]]
        else:
            # no definite choice, use classifier if available on ambiguous entries
            if model_tuple is not None and len(ambigs) > 0:
                model, preprocess, device = model_tuple
                best_idx = None
                best_score = 0.0
                best_label = None
                # run classifier on each ambiguous option and pick top marked
                for (idx, c, fr, patch) in ambigs:
                    cls, conf = classifier_predict(model, preprocess, device, patch)
                    # cls==1 -> marked
                    if cls == 1 and conf > best_score:
                        best_score = conf
                        best_idx = idx
                        best_label = option_letters[idx]
                        # store classifier decision into bubble_results (find by center)
                        # we will update bubble_results after q_no assignment
                if best_label is not None and best_score > 0.6:
                    detected_option = best_label
                else:
                    ambiguous_flag = True
                    detected_option = None
            else:
                # fallback: choose max fill ratio if it's moderately high
                if len(fill_list) > 0:
                    idx_max = max(range(len(fill_list)), key=lambda i: fill_list[i][1])
                    if fill_list[idx_max][1] > 0.12:
                        detected_option = option_letters[idx_max]
                    else:
                        ambiguous_flag = True
                        detected_option = None
        # assign qno and write detected_answers
        q_no = q_counter
        q_counter += 1
        key = str(q_no)
        detected_answers[key] = detected_option  # could be None
        # update corresponding bubble_results entries for this row to have q_no
        # find last len(row_sorted) entries and update (since we appended in order)
        # safer approach: update by matching centers
        for br in bubble_results:
            if br["q_no"] is None:
                # find matching center in row_sorted
                for idx, c in enumerate(row_sorted):
                    if br["center"][0] == int(c[0]) and br["center"][1] == int(c[1]):
                        br["q_no"] = key
                        # if this option is the detected_option mark chosen
                        if detected_option == br["option"]:
                            br["final_status"] = "selected"
                        else:
                            br["final_status"] = br.get("status", "unknown")
                        break
    # Build overlay visualization
    overlay = warped_rgb.copy()
    # color selected bubbles green, wrong red later when answer key provided, ambiguous yellow, others grey
    for b in bubble_results:
        x,y = int(b["center"][0]), int(b["center"][1])
        r = int(b["radius"])
        st = b.get("final_status", b.get("status", "empty"))
        color = (128,128,128)
        if st == "selected":
            color = (0,200,0)  # green
        elif st == "marked" or st == "ambiguous":
            color = (0,255,255)  # yellow
        elif st == "empty":
            color = (100,100,100)
        cv2.circle(overlay, (x,y), max(6, r), color, 2)
    # if answer key provided compute score and mark wrong ones red
    result = {
        "metadata": {
            "source_file": path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sheet_detected": bool(detected_sheet),
            "n_rows": len(rows),
            "n_bubbles": len(bubble_results)
        },
        "bubbles": bubble_results,
        "detected_answers": detected_answers
    }
    if answer_key is not None:
        scoring = compute_score(detected_answers, answer_key)
        result['scoring'] = scoring
        # mark wrongs in overlay: find per_question entries with status incorrect and color them red
        for qstr, info in scoring['per_question'].items():
            if info['status'] == 'incorrect':
                # get detected option center for this question and draw red cross
                det = info['detected']
                if det is not None:
                    # find bubble result with q_no==qstr and option==det
                    for b in bubble_results:
                        if b['q_no'] == qstr and b['option'] == det:
                            x,y = int(b['center'][0]), int(b['center'][1])
                            cv2.line(overlay, (x-10,y-10), (x+10,y+10), (0,0,255), 2)
                            cv2.line(overlay, (x-10,y+10), (x+10,y-10), (0,0,255), 2)
    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    overlay_path = os.path.join(out_dir, f"{basename}_overlay.png")
    json_path = os.path.join(out_dir, f"{basename}_result.json")
    cv2.imwrite(overlay_path, overlay)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    if debug:
        print(f"Processed {path} -> overlay: {overlay_path}, json: {json_path}")
    return overlay_path, json_path

# ---------------------------
# CLI & batch processing
# ---------------------------
EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.pdf'}

def find_image_files(input_dir: Path) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(str(input_dir)):
        for fn in filenames:
            if Path(fn).suffix.lower() in EXTS:
                files.append(str(Path(root) / fn))
    return sorted(files)

def collect_inputs(single: Optional[str], directory: Optional[str]) -> List[str]:
    if single:
        p = Path(single)
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {single}")
        if p.suffix.lower() not in EXTS:
            raise ValueError(f"Unsupported file: {p.suffix}")
        return [str(p)]
    if directory:
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(directory)
        return find_image_files(d)
    # Auto: attempt to find typical subfolders (Set_A / Set A / uploads)
    guesses = ['Set_A','Set A','uploads','input','data','Set_B','Set B']
    picked: List[str] = []
    for g in guesses:
        cand = DATASET_ROOT / g
        if cand.exists() and cand.is_dir():
            picked.extend(find_image_files(cand))
    if not picked:
        # fallback: root scan
        picked = find_image_files(DATASET_ROOT)
    return picked

def main():
    parser = argparse.ArgumentParser(description="OMR Pipeline - main.py (auto dataset mode)")
    parser.add_argument("--input_dir", help="Folder with images/PDFs (optional, else auto-detect)")
    parser.add_argument("--input", help="Single file to process instead of directory", default=None)
    parser.add_argument("--out_dir", help="Output folder (default: <dataset_root>/out)")
    parser.add_argument("--model_path", default=None, help="Optional classifier model (.pth). Use with --use_classifier flag")
    parser.add_argument("--use_classifier", action='store_true', help="Use classifier for ambiguous bubbles")
    parser.add_argument("--answer_key", default=None, help="JSON file with answer key (auto-discover if omitted)")
    parser.add_argument("--excel_key", default=None, help="Excel answer key file (supports sheet or column modes)")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    model_tuple = None
    if args.use_classifier:
        if args.model_path is None:
            print("Error: --use_classifier requires --model_path to be provided.")
            sys.exit(1)
        if not TORCH_AVAILABLE:
            print("Torch not installed or unavailable. Please install torch to use classifier.")
            sys.exit(1)
        model_tuple = load_classifier(args.model_path, device='cpu')

    # Determine output directory
    out_dir = args.out_dir or str(DATASET_ROOT / 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Load / auto-detect answer key
    answer_key = None
    excel_answer_sets = None
    if args.answer_key:
        answer_key = load_answer_key(args.answer_key)
    else:
        key_path = auto_key_file(DATASET_ROOT)
        if key_path:
            try:
                answer_key = load_answer_key(str(key_path))
                if args.debug:
                    print(f"[INFO] Loaded answer key: {key_path}")
            except Exception as ex:
                print(f"[WARN] Failed to load auto key: {ex}")

    # Excel key precedence if provided
    if args.excel_key:
        excel_answer_sets = load_answer_keys_from_excel(args.excel_key)
    else:
        # auto-discover excel key near dataset root
        for candidate in ['Key (Set A and B).xlsx','key (set a and b).xlsx','key.xlsx','Key.xlsx']:
            cand = DATASET_ROOT / candidate
            if cand.exists():
                excel_answer_sets = load_answer_keys_from_excel(str(cand))
                break

    # Collect files
    try:
        files = collect_inputs(args.input, args.input_dir)
    except Exception as ex:
        print(f"Error collecting inputs: {ex}")
        sys.exit(1)
    if not files:
        print("No images found to process.")
        sys.exit(1)

    print(f"[INFO] Dataset root: {DATASET_ROOT}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Files to process: {len(files)}")
    if excel_answer_sets is not None:
        # show summary sizes
        sizes = {k: len(v) for k,v in excel_answer_sets.items() if k != 'GLOBAL'}
        print(f"[INFO] Excel answer sets loaded: {sizes} (+GLOBAL={len(excel_answer_sets.get('GLOBAL',{}))})")
    elif answer_key is not None:
        print(f"[INFO] JSON answer key loaded: {len(answer_key)} questions")
    else:
        print("[INFO] No answer key provided/loaded. Scoring disabled.")

    processed = 0
    for f in tqdm(files):
        try:
            # dynamic answer key selection per image if excel sets available
            dyn_key = answer_key
            detected_letter = None
            used_key_label = None
            if excel_answer_sets is not None:
                # read minimal image (first page if PDF) for set OCR; reuse read_image logic partially
                try:
                    base_img = read_image(f)
                    detected_letter = extract_set_code(base_img)
                except Exception:
                    detected_letter = None
                logical = None
                if detected_letter == 'A':
                    logical = 'Set_A'
                elif detected_letter == 'B':
                    logical = 'Set_B'
                if logical is not None:
                    key_col = _select_excel_key(excel_answer_sets, logical)
                    if key_col:
                        dyn_key = {q: ans_list[0].upper() if len(ans_list)==1 else ans_list[0].upper() for q, ans_list in excel_answer_sets[key_col].items()}
                        used_key_label = key_col
                if dyn_key is None:
                    # fallback: if only one set present use it
                    if excel_answer_sets:
                        first = next(iter(excel_answer_sets.values()))
                        dyn_key = {q: v[0].upper() if isinstance(v, list) else v for q,v in first.items()}
                        used_key_label = next(iter(excel_answer_sets.keys()))
            ov_path, js_path = process_single_image(f, out_dir, model_tuple=model_tuple, answer_key=dyn_key, debug=args.debug)
            # augment JSON with set detection details if available
            if detected_letter is not None or used_key_label is not None:
                try:
                    with open(js_path,'r') as rf:
                        data = json.load(rf)
                    data['metadata']['detected_set_code'] = detected_letter
                    data['metadata']['used_answer_key_set'] = used_key_label
                    with open(js_path,'w') as wf:
                        json.dump(data, wf, indent=2)
                except Exception as e:
                    if args.debug:
                        print(f"[WARN] Failed to append set metadata: {e}")
            processed += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"[DONE] Processed {processed}/{len(files)} files -> {out_dir}")

if __name__ == "__main__":
    main()
