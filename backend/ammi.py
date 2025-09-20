"""
ammi.py
Offline helpers:
 - OfflineLayoutAnalyzer (OCR + optional local llama.cpp) for set / subject headers
 - Tiny CNN (train + infer) for ambiguous bubble fill classification

Train once:  python ammi.py  (creates bubble_cnn.onnx)
"""

import os, re, cv2, json, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ---------- OCR / LLaMA ----------
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

SUBJECT_CANON = [
    "PYTHON","DATA ANALYSIS","MYSQL","POWER BI","ADV STATS","STATISTICS","STATISTCS","EDA","SQL"
]

def canonical_subject(s: str) -> str:
    t = s.upper().strip()
    repl = {
        "DATAANALYSIS":"DATA ANALYSIS",
        "POWERBI":"POWER BI",
        "ADVSTATS":"ADV STATS",
        "STATISTICS":"ADV STATS",
        "STATISTCS":"ADV STATS",
        "EDA":"DATA ANALYSIS"
    }
    k = t.replace(" ","")
    for k0,v in repl.items():
        if k == k0.replace(" ",""):
            return v
    if t in SUBJECT_CANON: return t
    return t

@dataclass
class TextBox:
    text: str
    x: int
    y: int
    w: int
    h: int

class OfflineLayoutAnalyzer:
    def __init__(self, llama_model_path: Optional[str] = None, max_ctx: int = 2048):
        self.use_llama = False
        path = llama_model_path or os.getenv("OFFLINE_LLM_MODEL")
        if path and os.path.exists(path) and Llama is not None:
            try:
                self.llm = Llama(model_path=path, n_ctx=max_ctx, embeddings=False)
                self.use_llama = True
            except Exception:
                self.llm = None
        else:
            self.llm = None

    def ocr_boxes(self, img):
        if pytesseract is None:
            return []
        data = pytesseract.image_to_data(img, output_type='dict')
        boxes = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt: continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append(TextBox(txt, x, y, w, h))
        return boxes

    def heuristic_subject_spans(self, boxes: List[TextBox], img_w: int) -> List[Dict[str,int]]:
        if not boxes: return []
        top_limit = sorted([b.y + b.h for b in boxes])[max(0, int(len(boxes)*0.2))-1]
        top_boxes = [b for b in boxes if b.y < top_limit]
        cand = []
        for b in top_boxes:
            word = re.sub(r'[^A-Za-z]','', b.text)
            if not word: continue
            up = word.upper()
            if any(up in s.replace(" ","") or s.replace(" ","") in up for s in SUBJECT_CANON):
                cand.append(b)
        cand.sort(key=lambda b: b.x)
        merged = []
        for b in cand:
            if not merged:
                merged.append([b]); continue
            last_group = merged[-1]
            last_box = last_group[-1]
            if b.x - (last_box.x + last_box.w) < 30:
                last_group.append(b)
            else:
                merged.append([b])
        spans = []
        for g in merged:
            text = " ".join(bb.text for bb in g)
            canon = canonical_subject(text)
            xs = min(bb.x for bb in g)
            xe = max(bb.x + bb.w for bb in g)
            spans.append({"name": canon, "x_start": xs, "x_end": xe})
        final = []
        used = set()
        for s in spans:
            k = (s["name"], s["x_start"]//10, s["x_end"]//10)
            if k not in used:
                used.add(k); final.append(s)
        return final

    def detect_set_code(self, boxes: List[TextBox]) -> Optional[str]:
        for b in boxes:
            m = re.search(r"SET[^A-Z0-9]*([A-Z])", b.text.upper())
            if m: return f"Set_{m.group(1)}"
        for b in sorted(boxes, key=lambda b: b.y)[:15]:
            if re.fullmatch(r'[A-Z]', b.text.upper()):
                return f"Set_{b.text.upper()}"
        return None

    def llama_layout(self, boxes: List[TextBox]) -> Dict[str,Any]:
        if not self.use_llama: return {}
        lines = sorted(boxes, key=lambda b:(b.y,b.x))
        groups = []
        cur_y = None
        buf = []
        for b in lines:
            if cur_y is None or abs(b.y - cur_y) < 14:
                buf.append(b); cur_y = b.y
            else:
                groups.append(" ".join(bb.text for bb in buf))
                buf=[b]; cur_y=b.y
        if buf: groups.append(" ".join(bb.text for bb in buf))
        prompt = (
            "You are given OCR lines from an OMR sheet. Extract set code like Set_A (if any) "
            "and subject headers among PYTHON, DATA ANALYSIS, MYSQL, POWER BI, ADV STATS. "
            "Return JSON {\"set_code\": str|null, \"subjects\":[\"...\"]} only.\nLines:\n" +
            "\n".join(groups)
        )
        try:
            out = self.llm(prompt, max_tokens=256, temperature=0.1)
            txt = out["choices"][0]["text"]
            a = txt.find("{"); b = txt.rfind("}")
            if a!=-1 and b!=-1:
                return json.loads(txt[a:b+1])
        except Exception:
            pass
        return {}

    def analyze(self, img_bgr):
        boxes = self.ocr_boxes(img_bgr)
        if not boxes:
            return {"set_code": None, "subjects": [], "notes":"no_ocr"}
        set_code = self.detect_set_code(boxes)
        spans = self.heuristic_subject_spans(boxes, img_bgr.shape[1])
        llm_struct = self.llama_layout(boxes) if self.use_llama else {}
        llm_subs = {canonical_subject(s) for s in llm_struct.get("subjects", [])}
        if llm_subs:
            spans = [s for s in spans if s["name"] in llm_subs] or spans
        if llm_struct.get("set_code") and not set_code:
            set_code = llm_struct["set_code"]
        return {
            "set_code": set_code,
            "subjects": spans,
            "notes": f"ocr_boxes={len(boxes)} llama_used={self.use_llama}"
        }

# ---------- Tiny CNN (train + infer) ----------
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import onnxruntime as ort

class BubbleDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, cls in enumerate(["empty","filled"]):
            d = os.path.join(root, cls)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                if f.lower().endswith((".png",".jpg",".jpeg")):
                    self.samples.append((os.path.join(d,f), label))
        self.tx = transforms.ToTensor()
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,l = self.samples[i]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None: g = np.zeros((32,32), np.uint8)
        g = cv2.resize(g,(32,32))
        return self.tx(g), torch.tensor(l)

class TinyBubbleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1, groups=16), nn.Conv2d(16,32,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64,2)
    def forward(self,x):
        x = self.net(x).view(x.size(0), -1)
        return self.fc(x)

def train_bubble_cnn(root="dataset_bubbles", epochs=8, batch=64, lr=1e-3, out="bubble_cnn.onnx"):
    ds = BubbleDataset(root)
    if len(ds)==0:
        print("No samples in dataset_bubbles."); return
    val = max(1,int(0.1*len(ds)))
    tr_idx = list(range(len(ds)-val))
    va_idx = list(range(len(ds)-val, len(ds)))
    dl_tr = DataLoader(torch.utils.data.Subset(ds,tr_idx), batch_size=batch, shuffle=True)
    dl_va = DataLoader(torch.utils.data.Subset(ds,va_idx), batch_size=batch)
    model = TinyBubbleCNN()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0
    for ep in range(1, epochs+1):
        model.train(); run=0
        for x,y in dl_tr:
            opt.zero_grad()
            outp = model(x)
            loss = crit(outp,y)
            loss.backward(); opt.step()
            run += loss.item()*x.size(0)
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in dl_va:
                p = model(x)
                pred = p.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        acc = correct/total if total else 0
        print(f"Epoch {ep}: loss={run/len(tr_idx):.4f} val_acc={acc:.3f}")
        if acc > best:
            best = acc
            dummy = torch.randn(1,1,32,32)
            torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"], opset_version=11)
            print(f"Saved {out} (best_acc={best:.3f})")
    print("Done.")

class BubbleCNNInfer:
    def __init__(self, model_path="bubble_cnn.onnx", thresh=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0].name
        self.thresh = thresh
    def predict_filled(self, patch_bgr):
        g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g,(32,32)).astype(np.float32)/255.0
        g = g[None,None,...]
        logits = self.session.run(None,{self.inp:g})[0]
        probs = self._softmax(logits)[0]
        return probs[1] >= self.thresh, float(probs[1])
    def _softmax(self,x):
        e = np.exp(x - x.max(axis=1,keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    # Train if run directly
    train_bubble_cnn()