import os
import cv2
import numpy as np
import json
import fitz  # PyMuPDF
import pandas as pd  # <-- NEW IMPORT
try:
    import pytesseract  # OCR (optional)
    # Allow environment variable override for Windows install path
    _tess_cmd = os.environ.get('TESSERACT_CMD')
    if _tess_cmd and os.path.exists(_tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except Exception:
    pytesseract = None
from pathlib import Path
from typing import Dict as _Dict
import cv2
import numpy as np
import pytesseract
import re


# --- CONFIGURATION & CONSTANTS ---

# Automatic dataset discovery.
# Tries these relative candidates (in order) and picks the first containing a set folder:
#   ../dataset/data
#   ../dataset/Theme 1 - Sample Data
#   ../dataset
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE.parent / 'dataset' / 'data',
    _HERE.parent / 'dataset' / 'Theme 1 - Sample Data',
    _HERE.parent / 'dataset'
]

def _select_dataset_root() -> Path:
    for cand in _CANDIDATES:
        if not cand.exists():
            continue
        names = {p.name.lower() for p in cand.iterdir() if p.is_dir()}
        if any(n in names for n in ['set_a','set a','set_b','set b']):
            return cand
    return _CANDIDATES[0]

DATASET_ROOT = _select_dataset_root()
OUTPUT_DIR = str(DATASET_ROOT / 'results')

def _find_key_file(root: Path) -> Path:
    patterns = [
        'Key (Set A and B).xlsx', 'key (set a and b).xlsx', 'key.xlsx', 'Key.xlsx'
    ]
    for name in patterns:
        c = root / name
        if c.exists():
            return c
    for f in root.glob('*.xlsx'):
        lname = f.name.lower()
        if 'key' in lname and 'a' in lname and 'b' in lname:
            return f
    raise FileNotFoundError('Answer key Excel not found in dataset root.')

try:
    KEY_FILE_PATH = str(_find_key_file(DATASET_ROOT))
except Exception:
    KEY_FILE_PATH = ''  # handled later

def _normalize_set(name: str) -> str:
    cleaned = name.replace('-', ' ').replace('_', ' ').strip().lower()
    if cleaned in {'set a','a'}:
        return 'Set_A'
    if cleaned in {'set b','b'}:
        return 'Set_B'
    return name

def discover_sets(root: Path) -> _Dict[str, Path]:
    mapping: _Dict[str, Path] = {}
    if not root.exists():
        return mapping
    for p in root.iterdir():
        if p.is_dir():
            norm = _normalize_set(p.name)
            if norm in {'Set_A','Set_B'}:
                mapping[norm] = p
    return mapping

SET_FOLDERS = discover_sets(DATASET_ROOT)

print('[INFO] Dataset root:', DATASET_ROOT)
print('[INFO] Key file:', KEY_FILE_PATH or 'NOT FOUND')
print('[INFO] Sets found:', {k: str(v) for k,v in SET_FOLDERS.items()})

# 2. OMR Sheet Layout Configuration
OMR_CONFIG = {
    'TOTAL_QUESTIONS': 100,
    'CHOICES_PER_QUESTION': 5,
    'QUESTIONS_PER_COLUMN': 20,
    'COLUMNS_PER_SHEET': 5,
    'X_START_OFFSET': 120,
    'Y_START_OFFSET': 230,
    'X_SPACING': 55,
    'Y_SPACING': 27.5,
    'BUBBLE_RADIUS': 10,
    'BUBBLE_THRESHOLD': 100,
    'HORIZONTAL_SEARCH_RADIUS': 8  # pixels left/right search to refine bubble center
}


# --- HELPER & DYNAMIC KEY LOADING FUNCTIONS ---

def load_answer_keys_from_excel(path):
    """Load answer keys supporting two formats:

    1. Sheet-based: Workbook has separate sheets e.g. 'Set-A', 'Set-B' each containing a single column (or multiple columns) of lines like '1 - a'. We parse each sheet into a key.
    2. Column-based (fallback): Single sheet with multiple columns (prior implementation).

    Returns dict where keys are normalized sheet names ('Set_A','Set_B', etc.) or original column names plus optional 'GLOBAL'.
    """
    def _parse_series(series, global_key):
        col_key: dict[str, list[str]] = {}
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
            col_key[qno] = answers
            if qno not in global_key:
                global_key[qno] = answers
        return col_key

    try:
        xls = pd.ExcelFile(path)
    except FileNotFoundError:
        print(f"Error: Answer key file not found at {path}")
        return None
    except Exception as e:
        print(f"An error occurred while opening the Excel file: {e}")
        return None

    sheet_names_lower = [s.lower() for s in xls.sheet_names]
    sheet_mode = any('set' in s for s in sheet_names_lower)  # heuristic

    answer_keys: dict[str, dict[str, list[str]]] = {}
    global_key: dict[str, list[str]] = {}

    if sheet_mode:
        print('[INFO] Parsing answer keys in sheet-based mode.')
        for sheet in xls.sheet_names:
            norm = _normalize_set(sheet)
            try:
                df_sheet = xls.parse(sheet, header=None)
            except Exception as e:
                print(f"[WARN] Failed parsing sheet {sheet}: {e}")
                continue
            # Flatten all columns of the sheet into a single series
            series = pd.Series([v for v in df_sheet.fillna('').values.flatten() if str(v).strip()])
            parsed = _parse_series(series, global_key)
            if parsed:
                answer_keys[norm] = parsed
        if not answer_keys:
            print('[WARN] Sheet-based detection yielded no keys, falling back to column-based first sheet parsing.')
            sheet_mode = False  # fall through

    if not sheet_mode:
        try:
            df = xls.parse(xls.sheet_names[0], header=0)
        except Exception as e:
            print(f"[ERROR] Could not parse first sheet for column-based mode: {e}")
            return None
        for col in df.columns:
            series = df[col]
            parsed = _parse_series(series, global_key)
            if parsed:
                answer_keys[col] = parsed

    if global_key:
        answer_keys['GLOBAL'] = global_key

    if not answer_keys:
        print('Error: No valid question/answer patterns parsed from Excel.')
        return None

    print(f"Successfully loaded answer keys ({'sheet' if sheet_mode else 'column'} mode): {list(answer_keys.keys())}")
    return answer_keys

def order_points(pts):
    """Orders 4 points of a contour in a consistent sequence."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# --- CORE CV PIPELINE (Functions remain the same) ---

def transform_perspective(image, width=800, height=1000):
    """Finds the OMR sheet in an image and returns a top-down view."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None

    ordered_pts = order_points(screenCnt.reshape(4, 2))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def read_bubbles(warped_image, config):
    """Reads marked bubbles using adaptive thresholding with horizontal search refinement.

    For each predicted bubble center we search a small horizontal window to find
    the offset giving maximal ink (non-zero pixels) which mitigates systematic
    left/right shifts that previously caused A/B misclassification.
    """
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    answers = {}
    h_search = config.get('HORIZONTAL_SEARCH_RADIUS', 0)
    for col in range(config['COLUMNS_PER_SHEET']):
        for row in range(config['QUESTIONS_PER_COLUMN']):
            q_num = col * config['QUESTIONS_PER_COLUMN'] + row + 1
            bubble_scores = []
            for choice_idx in range(config['CHOICES_PER_QUESTION']):
                base_cx = int(config['X_START_OFFSET'] + (col * config['CHOICES_PER_QUESTION'] + choice_idx) * config['X_SPACING'])
                cy = int(config['Y_START_OFFSET'] + row * config['Y_SPACING'])
                best_score = -1
                # horizontal refinement
                for offset in range(-h_search, h_search + 1, max(1, h_search // 4) if h_search else 1):
                    cx = base_cx + offset
                    if cx < 0 or cx >= thresh.shape[1]:
                        continue
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.circle(mask, (cx, cy), config['BUBBLE_RADIUS'], 255, -1)
                    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total_pixels = cv2.countNonZero(masked)
                    if total_pixels > best_score:
                        best_score = total_pixels
                bubble_scores.append(best_score)
            if max(bubble_scores) > config['BUBBLE_THRESHOLD']:
                answers[str(q_num)] = chr(ord('A') + int(np.argmax(bubble_scores)))
            else:
                answers[str(q_num)] = "NONE"
    return answers


# --- OCR: SET CODE EXTRACTION ---
def extract_set_code(image) -> str | None:
    """Attempt to OCR the 'Set No' field from the raw (pre-warp) image.

    Strategy:
      1. Crop a band from the top 20% of the image where header resides.
      2. Convert to grayscale, apply mild threshold to enhance text.
      3. Use pytesseract (if available) to extract text.
      4. Search for patterns like 'set', 'set no', 'set:' followed by a single letter (A/B/C...).

    Returns uppercase single letter if found (e.g. 'A'), else None.
    """
    if pytesseract is None:
        return None
    try:
        h, w = image.shape[:2]
        header = image[0:int(h*0.25), :]
        gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
        # Light preprocessing: blur then adaptive threshold (invert) to help OCR
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Some images may have colored pen; keep original too
        ocr_img = thresh
        config = '--psm 6'  # assume a block of text
        text = pytesseract.image_to_string(ocr_img, config=config)
        if not text:
            return None
        text_lower = text.lower().replace('\n', ' ')
        # Patterns: 'set no', 'set :' 'set -', 'set'
        import re
        match = re.search(r'set\s*(no\.?:|number:|:)??\s*([a-z])', text_lower)
        if match:
            letter = match.group(2).upper()
            if letter.isalpha():
                return letter
        # Fallback: look for isolated single letter after the word 'set'
        match2 = re.search(r'set[^a-z0-9]{1,5}([a-z])\b', text_lower)
        if match2:
            letter = match2.group(1).upper()
            return letter
    except Exception:
        return None
    return None

# --- OCR SUBJECT BLOCKS & BUBBLE ANALYSIS (User-provided logic integrated) ---
def find_subject_blocks(image):
    """
    Finds subject titles via OCR and defines their bounding boxes.
    Returns: (subject_blocks: dict[name]->(x_start, y_start, x_end, y_end), ocr_data)
    """
    if pytesseract is None:
        return {}, None
    # Use pytesseract to get all text data
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    subject_blocks = {}
    # Define your subject keywords here (extendable)
    SUBJECT_KEYWORDS = [
        "PYTHON", "DATA ANALYSIS", "MYSQL", "POWER BI", "ADV STATS",
        "STATISTICS", "MACHINE LEARNING", "DEEP LEARNING"
    ]

    # First, find the coordinates of all subject titles
    found_subjects = []
    n = len(ocr_data.get('text', [])) if ocr_data else 0
    for i in range(n):
        text = ocr_data['text'][i]
        if not text:
            continue
        text_up = text.upper()
        if any(keyword in text_up for keyword in SUBJECT_KEYWORDS):
            # Find the actual keyword that matched
            matched_keyword = next((keyword for keyword in SUBJECT_KEYWORDS if keyword in text_up), text_up)
            x = int(ocr_data['left'][i]); y = int(ocr_data['top'][i])
            w = int(ocr_data['width'][i]); h = int(ocr_data['height'][i])
            found_subjects.append({'keyword': matched_keyword, 'y': y, 'x': x})

    # Sort subjects by their vertical position on the page
    found_subjects.sort(key=lambda s: s['y'])

    # Define the block for each subject
    image_height, image_width = image.shape[:2]
    for i, subject in enumerate(found_subjects):
        x_start = 0  # Assume subject blocks span the full width
        y_start = subject['y']
        y_end = found_subjects[i+1]['y'] if i + 1 < len(found_subjects) else image_height
        subject_blocks[subject['keyword']] = (x_start, y_start, image_width, y_end)

    # print debug
    if subject_blocks:
        print(f"Found subject blocks: {list(subject_blocks.keys())}")
    return subject_blocks, ocr_data

def _build_structured_data_from_ocr(ocr_data, subject_blocks):
    """Heuristic builder that maps questions to option text boxes using OCR.
    Returns: structured_data like { q_num: { 'options': { 'A': {'box': (x,y,w,h)}, ... } } }
    Notes:
      - This is a best-effort approximation. It uses option header letters (A,B,C,D) in the subject area
        and pairs them with the Y of detected question numbers within that subject area.
    """
    if not ocr_data or not subject_blocks:
        return {}
    # Gather OCR entries
    entries = []
    n = len(ocr_data.get('text', []))
    for i in range(n):
        txt = str(ocr_data['text'][i]).strip()
        if not txt:
            continue
        try:
            x = int(ocr_data['left'][i]); y = int(ocr_data['top'][i])
            w = int(ocr_data['width'][i]); h = int(ocr_data['height'][i])
        except Exception:
            continue
        entries.append({"text": txt, "x": x, "y": y, "w": w, "h": h})

    # Helper to check if a point is within a block
    def in_block(e, block):
        x1, y1, x2, y2 = block
        return (e['x'] >= x1 and e['x'] <= x2 and e['y'] >= y1 and e['y'] <= y2)

    structured = {}
    for subj, block in subject_blocks.items():
        # Option headers in block (top-most A/B/C/D)
        opts = {o: [] for o in ['A','B','C','D']}
        nums = []  # question numbers (text purely digits)
        for e in entries:
            if not in_block(e, block):
                continue
            t = e['text'].upper()
            if t in opts:
                opts[t].append(e)
            elif t.isdigit():
                qn = int(t)
                if 1 <= qn <= 100:
                    nums.append((qn, e))
        # Choose top-most for each option letter as header reference
        header = {}
        for k, lst in opts.items():
            if lst:
                header[k] = sorted(lst, key=lambda d: d['y'])[0]
        # For each question number in this subject, assign option boxes based on header x and this q's y
        for (qno, e) in sorted(nums, key=lambda p: p[0]):
            options = {}
            for k in ['A','B','C','D']:
                if k in header:
                    hx, hy, hw, hh = header[k]['x'], header[k]['y'], header[k]['w'], header[k]['h']
                    # create a pseudo box at same x position but align y to this question's number top
                    options[k] = { 'box': (hx, e['y'], hw, hh) }
            if options:
                structured[str(qno)] = { 'options': options, 'subject': subj }
    return structured

def analyze_bubbles_below_text(thresholded_image, structured_data):
    """
    MODIFIED: Analyzes bubbles located BELOW the OCR'd option text.
    """
    # The bubble is BELOW the letter, so we use a POSITIVE Y offset
    # **TWEAK THESE VALUES** for perfect alignment
    Y_OFFSET = 30  # Pixels BELOW the option text's bottom edge
    X_OFFSET = 5   # Small horizontal adjustment
    BUBBLE_RADIUS = 8
    BUBBLE_THRESHOLD = 80 # Min dark pixels

    results = {}
    for q_num, data in structured_data.items():
        results[q_num] = {}
        attempted_option = None
        max_pixels = -1

        for option, opt_data in data['options'].items():
            x, y, w, h = opt_data['box']
            # Calculate the center of the bubble below the text
            cx = x + (w // 2) + X_OFFSET
            cy = y + h + Y_OFFSET # Use bottom of text (y+h) as reference
            mask = np.zeros(thresholded_image.shape, dtype="uint8")
            cv2.circle(mask, (cx, cy), BUBBLE_RADIUS, 255, -1)
            mask = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)
            total_pixels = cv2.countNonZero(mask)

            if total_pixels > max_pixels:
                max_pixels = total_pixels
                if total_pixels > BUBBLE_THRESHOLD:
                    attempted_option = option
        for option in ['A', 'B', 'C', 'D']:
            if option in data['options']:
                if option == attempted_option:
                    results[q_num][option] = "Attempted"
                else:
                    results[q_num][option] = "Not Attempted"
    return results

def _threshold_for_ocr_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 12)
    return thr


# --- RESULT PROCESSING (Function remains the same) ---

def _extract_subject_titles(warped_image: np.ndarray) -> list[str]:
    """OCR top region to extract subject column titles in order.

    Heuristic: Crop a horizontal band where headers appear, run pytesseract, collect
    distinct words/phrases that match expected pattern (letters & spaces) and appear
    above first question row (using Y_START_OFFSET).
    """
    titles: list[str] = []
    if pytesseract is None:
        return titles
    h, w = warped_image.shape[:2]
    band_top = 0
    band_bottom = int(OMR_CONFIG['Y_START_OFFSET'] * 0.9)
    band = warped_image[band_top:band_bottom, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # light threshold for better text contrast
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT)
    collected = []
    for i, txt in enumerate(data.get('text', [])):
        if not txt:
            continue
        clean = re.sub(r'[^A-Za-z ]', '', txt).strip()
        if len(clean) < 2:
            continue
        x = data['left'][i]; y = data['top'][i]
        # discard if y too low (safety)
        if y > band_bottom * 0.9:
            continue
        collected.append((x, clean.upper()))
    # sort by x, keep unique order preserving
    collected.sort(key=lambda t: t[0])
    seen = set()
    for _, word in collected:
        if word not in seen:
            seen.add(word)
            titles.append(word)
    # heuristic: keep first 5 at most (PYTHON, DATA ANALYSIS, MYSQL, POWER BI, ADV STATS)
    return titles[:OMR_CONFIG['COLUMNS_PER_SHEET']]

def generate_json_report(image_path, student_answers, answer_key, exam_set, subject_titles=None):
    """Score sheet and produce JSON serializable report with optional dynamic subject titles."""
    total_score = 0
    if subject_titles:
        subject_scores = {title: 0 for title in subject_titles}
    else:
        subject_scores = {f"subject_{i+1}": 0 for i in range(OMR_CONFIG['COLUMNS_PER_SHEET'])}
    summary = {"correct": 0, "incorrect": 0, "unattempted": 0}
    details = []
    for q_num_str, correct_list in answer_key.items():
        if isinstance(correct_list, str):
            correct_list_norm = [correct_list.lower()]
        else:
            correct_list_norm = [c.lower() for c in correct_list]
        marked_answer = student_answers.get(q_num_str)
        if not marked_answer or marked_answer == "NONE":
            summary['unattempted'] += 1
            status = 'Unattempted'
        else:
            m = marked_answer.lower()
            if m in correct_list_norm:
                summary['correct'] += 1
                total_score += 1
                q_int = int(q_num_str)
                subject_idx = (q_int - 1) // OMR_CONFIG['QUESTIONS_PER_COLUMN']
                if subject_titles and subject_idx < len(subject_titles):
                    subject_scores[subject_titles[subject_idx]] += 1
                else:
                    subject_scores[f"subject_{subject_idx+1}"] += 1
                status = 'Correct'
            else:
                summary['incorrect'] += 1
                status = 'Incorrect'
        details.append({
            'question': int(q_num_str),
            'marked': marked_answer,
            'correct': correct_list_norm,
            'status': status
        })
    report = {
        'source_image': os.path.basename(image_path),
        'exam_set': exam_set,
        'total_score': total_score,
        'subject_scores': subject_scores,
        'summary': summary,
        'details': sorted(details, key=lambda x: x['question'])
    }
    return report

# --- MAIN DRIVER ---

if __name__ == "__main__":
    if not KEY_FILE_PATH:
        print('ERROR: No key Excel file located. Place it in the dataset root.')
        exit()

    ANSWER_KEYS = load_answer_keys_from_excel(KEY_FILE_PATH)
    if ANSWER_KEYS is None:
        print('Halting execution due to error in loading answer keys.')
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'[INFO] Output directory: {OUTPUT_DIR}')

    def _match_key_col(logical: str) -> str:
        variants = [logical, logical.replace('_',' '), logical.upper(), logical.replace('_',' ').title(), logical.replace('_','').title()]
        for v in variants:
            if v in ANSWER_KEYS:
                return v
        if 'GLOBAL' in ANSWER_KEYS:
            return 'GLOBAL'
        return ''

    def _logical_from_detected(letter: str | None) -> str | None:
        if not letter:
            return None
        letter = letter.upper().strip()
        if letter == 'A':
            return 'Set_A'
        if letter == 'B':
            return 'Set_B'
        return None

    # Iterate discovered sets
    total_files = 0
    total_reports = 0

    for logical, folder in SET_FOLDERS.items():
        key_col = _match_key_col(logical)
        if not key_col:
            print(f"[WARN] No matching answer key column and no GLOBAL fallback for '{logical}' -> skipping")
            continue
        print(f"\n--- Processing Exam Set: {logical} (key column used: {key_col}) ---")
        base_answer_key = ANSWER_KEYS[key_col]

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            images_to_process = []

            if filename.lower().endswith('.pdf'):
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        images_to_process.append((img_bgr, f"{filename}_page_{page_num+1}"))
                    print(f"Processed PDF: {filename}")
                except Exception as e:
                    print(f"Error processing PDF {filename}: {e}")
                    continue
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                if image is not None:
                    images_to_process.append((image, filename))
                else:
                    print(f"Could not read image: {filename}")
                    continue
            else:
                continue  # skip unsupported files

            total_files += len(images_to_process)
            for image, original_name in images_to_process:
                # Attempt to detect set code from raw image before warp
                detected_code = extract_set_code(image)
                detected_logical = _logical_from_detected(detected_code)
                warped = transform_perspective(image)
                if warped is None:
                    print(f"Failed to find OMR sheet in: {original_name}")
                    continue

                student_answers = read_bubbles(warped, OMR_CONFIG)
                # Optional: OCR-driven cross-check using subject titles and option text
                try:
                    subj_blocks, ocr_data = find_subject_blocks(warped)
                    if subj_blocks:
                        struct = _build_structured_data_from_ocr(ocr_data, subj_blocks)
                        thr = _threshold_for_ocr_analysis(warped)
                        ocr_attempts = analyze_bubbles_below_text(thr, struct)
                    else:
                        ocr_attempts = {}
                except Exception:
                    ocr_attempts = {}
                # If OCR found a different set than current loop logical, attempt to override key
                # start from base answer key each file
                answer_key = base_answer_key
                used_key_label = key_col
                effective_exam_set = logical
                if detected_logical and detected_logical != logical:
                    alt_key_col = _match_key_col(detected_logical)
                    if alt_key_col:
                        answer_key = ANSWER_KEYS[alt_key_col]
                        used_key_label = alt_key_col
                        effective_exam_set = detected_logical
                # Dynamic subject titles (from warped image header)
                try:
                    subject_titles = _extract_subject_titles(warped)
                except Exception:
                    subject_titles = None
                report = generate_json_report(original_name, student_answers, answer_key, effective_exam_set, subject_titles)
                # annotate set code info
                report['detected_set_code'] = detected_code
                report['used_answer_key_set'] = used_key_label
                if ocr_attempts:
                    report['ocr_subject_blocks_present'] = True
                    report['ocr_attempts_meta'] = {'n_questions': len(ocr_attempts)}
                else:
                    report['ocr_subject_blocks_present'] = False

                output_filename = os.path.splitext(original_name)[0] + '.json'
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=4)

                print(f"  -> Successfully generated report for {original_name} -> {output_filename}")
                total_reports += 1

    print(f"\n--- All processing complete. Processed {total_files} file pages -> {total_reports} reports. ---")