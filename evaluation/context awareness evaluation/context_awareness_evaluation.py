import json
import pandas as pd
import re

# =========================
# CONFIG
# =========================
INPUT_XLSX = "compare_answers_EN.xlsx"
OUTPUT_XLSX = "Model_Comparison_Result_EN.xlsx"

from trnlp import TrnlpWord
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# =========================
# LANG + STEM
# =========================
def is_turkish(text: str) -> bool:
    try:
        return detect(str(text)) == "tr"
    except Exception:
        return False

def turkish_stem_string(text: str) -> str:
    words = re.findall(r"\b\w+\b", str(text).lower())
    stems = []
    for w in words:
        tw = TrnlpWord()
        tw.setword(w)
        stem_attr = tw.get_stem
        stem = stem_attr() if callable(stem_attr) else stem_attr
        stems.append(str(stem) if stem else w)
    return " ".join(stems)

# =========================
# UTILS
# =========================
def norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if is_turkish(x):
        return turkish_stem_string(x).lower()
    return str(x).lower()

def norm_sys(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).lower()

def parse_system_info(cell):
    if isinstance(cell, dict):
        if "system_info" in cell and isinstance(cell["system_info"], dict):
            return cell["system_info"]
        return cell

    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}

    s = str(cell).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "system_info" in obj and isinstance(obj["system_info"], dict):
            return obj["system_info"]
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        snippet = s[l:r+1]
        try:
            return json.loads(snippet)
        except Exception:
            try:
                return json.loads(snippet.replace("'", '"'))
            except Exception:
                return {}

    return {}

def contains_any_part_sys(sys_value: str, answer_text: str, min_len: int = 3) -> bool:
    sys_value = norm_sys(sys_value)
    if not sys_value or not answer_text:
        return False
    parts = re.findall(r"[a-z0-9]+", sys_value)
    parts = [p for p in parts if len(p) >= min_len]
    return any(p in answer_text for p in parts)

# =========================
# KEYWORD SETS
# =========================
OS_WORDS = ["windows", "ubuntu", "linux", "macos", "ventura", "sonoma", "darwin"]
GPU_WORDS = ["nvidia", "amd", "radeon", "intel", "igpu", "apple silicon", "m1", "m2"]
NET_WORDS = [
    "vpn", "public", "home", "network", "dns", "proxy", "router", "ethernet",
    "firewall", "modem", "wifi", "wi-fi",
    "bağlantı", "internet", "ağ", "güvenlik duvarı", "yönlendirici",
    "kablosuz", "kablolu", "genel ağ", "özel ağ", "ev ağı", "ağ profili"
]
RAM_WORDS = ["ram", "gb", "memory", "bellek", "gb ram"]

ACTION_WORDS = [
    "disable", "enable", "turn off", "turn on",
    "restart", "reboot", "update", "upgrade",
    "reinstall", "install", "remove", "uninstall",
    "reset", "flush", "check", "verify",
    "change", "switch", "ddu", "clean install",
    "flush dns",
    "kapat", "kapatın", "kapatmayı", "başlat",
    "aç", "açın", "sil", "kur",
    "devre dışı", "devre dışı bırak",
    "etkinleştir", "aktif et",
    "güncelle", "güncelleyin",
    "yeniden başlat", "yeniden başlatın",
    "kaldır",
    "yeniden kur", "tekrar kur",
    "değiştir", "ayarlarını değiştir", "ayarları değiştir",
    "kontrol et", "kontrol edin",
    "sıfırla", "resetle"
]

# =========================
# STRICT MATCH HELPERS
# =========================
def extract_os_mentions(text: str) -> set:
    t = norm_sys(text)
    mentions = set()

    if "windows" in t:
        mentions.add("windows")
    if "ubuntu" in t or "linux" in t:
        mentions.add("linux")
    if "macos" in t or "darwin" in t or "ventura" in t or "sonoma" in t:
        mentions.add("macos")

    return mentions

def extract_gpu_mentions(text: str) -> set:
    t = norm_sys(text)
    mentions = set()

    if "nvidia" in t:
        mentions.add("nvidia")
    if "amd" in t or "radeon" in t:
        mentions.add("amd")
    if "intel" in t:
        mentions.add("intel")
    if "apple silicon" in t or "m1" in t or "m2" in t or "apple" in t:
        mentions.add("apple")

    return mentions

def extract_ram_mentions(text: str) -> set:
    t = norm_sys(text)
    vals = re.findall(r"\b(\d+)\s*gb\b", t)
    return set(vals)

def strict_single_match(answer_mentions: set, true_mentions: set) -> int:
    if not answer_mentions or not true_mentions:
        return 0
    if answer_mentions.isdisjoint(true_mentions):
        return 0
    if not answer_mentions.issubset(true_mentions):
        return 0
    return 1

def net_match_strict(answer_text: str, sys_net: str, sys_vpn: str) -> int:
    a = norm_sys(answer_text)

    contradiction = False
    positive = False

    net_true = set()
    if "home" in sys_net or "private" in sys_net or "ev" in sys_net or "özel" in sys_net:
        net_true.add("home")
    if "public" in sys_net or "genel" in sys_net:
        net_true.add("public")

    net_ans = set()
    if "home" in a or "private" in a or "ev ağı" in a or "özel ağ" in a:
        net_ans.add("home")
    if "public" in a or "genel ağ" in a:
        net_ans.add("public")

    if net_ans:
        positive = True
        if not net_true:
            contradiction = True
        elif net_ans.isdisjoint(net_true):
            contradiction = True
        elif not net_ans.issubset(net_true):
            contradiction = True

    vpn_true = None
    if "on" in sys_vpn:
        vpn_true = "on"
    elif "off" in sys_vpn:
        vpn_true = "off"

    vpn_ans = set()
    if "enable vpn" in a or "turn on vpn" in a or "vpn aç" in a or "vpn etkinleştir" in a:
        vpn_ans.add("on")
    if "disable vpn" in a or "turn off vpn" in a or "vpn kapat" in a or "vpn devre dışı" in a:
        vpn_ans.add("off")

    if vpn_ans:
        positive = True
        if vpn_true is None:
            contradiction = True
        elif vpn_true not in vpn_ans:
            contradiction = True
        elif len(vpn_ans) > 1:
            contradiction = True

    if contradiction:
        return 0

    return 1 if positive else 0

# =========================
# SCORING
# =========================
def score_answer(answer, system_info):
    a = norm(answer)

    # ref_score (0-4)
    ref_os  = any(w in a for w in OS_WORDS)
    ref_gpu = any(w in a for w in GPU_WORDS)
    ref_net = any(w in a for w in NET_WORDS)
    ref_ram = any(w in a for w in RAM_WORDS)
    ref_score = sum([ref_os, ref_gpu, ref_net, ref_ram])

    # action_score (0-2)
    action = any(w in a for w in ACTION_WORDS)
    action_score = 2 if action and ref_score > 0 else 1 if action else 0

    # system info
    sys = system_info if isinstance(system_info, dict) else {}

    sys_os = norm_sys(sys.get("os") or sys.get("system") or "")
    sys_gpu = norm_sys(sys.get("gpu") or "")
    sys_net = norm_sys(sys.get("network") or "")
    sys_vpn = norm_sys(sys.get("vpn") or "")
    sys_ram = norm_sys(
        sys.get("ram") or
        sys.get("memory") or
        sys.get("bellek") or
        sys.get("ram_gb") or
        ""
    )

    # sys_ref (0-4)
    sys_ref_os  = contains_any_part_sys(sys_os, a)
    sys_ref_gpu = contains_any_part_sys(sys_gpu, a)
    sys_ref_net = contains_any_part_sys(sys_net, a) or contains_any_part_sys(sys_vpn, a)
    sys_ref_ram = contains_any_part_sys(sys_ram, a, min_len=1)
    sys_ref = sum([sys_ref_os, sys_ref_gpu, sys_ref_net, sys_ref_ram])

    # sys_match (0-4)
    sys_match = 0

    if ref_os and sys_os:
        true_os = extract_os_mentions(sys_os)
        ans_os = extract_os_mentions(a)
        sys_match += strict_single_match(ans_os, true_os)

    if ref_gpu and sys_gpu:
        true_gpu = extract_gpu_mentions(sys_gpu)
        ans_gpu = extract_gpu_mentions(a)
        sys_match += strict_single_match(ans_gpu, true_gpu)

    if ref_net and (sys_net or sys_vpn):
        sys_match += net_match_strict(a, sys_net, sys_vpn)

    if ref_ram and sys_ram:
        true_ram = extract_ram_mentions(sys_ram)
        ans_ram = extract_ram_mentions(a)
        sys_match += strict_single_match(ans_ram, true_ram)

    return {
        "ref_score": ref_score,
        "action_score": action_score,
        "sys_ref": sys_ref,
        "sys_match": sys_match,
    }

# =========================
# MAIN
# =========================
df = pd.read_excel(INPUT_XLSX)

# İlk 6 sütunu al: A=query, B-E=methods, F=system_info
df = df.iloc[:, :7].copy()

query_col = df.columns[0]
method_cols = list(df.columns[1:6])   # B, C, D, E, F
system_col = df.columns[6]

df["sys"] = df[system_col].apply(parse_system_info)

score_frames = []
summary_rows = []

for method in method_cols:
    scores = df.apply(
        lambda r: score_answer(r[method], r["sys"]),
        axis=1,
        result_type="expand"
    )
    scores.columns = [f"{method}_ref_score", f"{method}_action_score", f"{method}_sys_ref", f"{method}_sys_match"]
    score_frames.append(scores)

    summary_rows.append({
        "method": method,
        "avg_ref_score": scores[f"{method}_ref_score"].mean(),
        "avg_action_score": scores[f"{method}_action_score"].mean(),
        "avg_sys_ref": scores[f"{method}_sys_ref"].mean(),
        "avg_sys_match": scores[f"{method}_sys_match"].mean(),
    })

out = pd.concat([df.drop(columns=["sys"])] + score_frames, axis=1)
summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="detailed_scores", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print("[OK] Written to", OUTPUT_XLSX)