#!/usr/bin/env python3
"""從 Obsidian vault 同步論文導讀到網站(papers/*.md)。

用法:python3 scripts/sync.py
- 讀 vault 的 00_主論文 筆記,剝除研究策略段落與私人措辭,輸出到 papers/。
- PATCHES 收錄歷次手術補丁;來源改版導致補丁失配時會警告(需人工複核)。
- 最後跑隱私掃描,命中未豁免的關鍵詞即以非零碼退出。
"""
import re, sys, pathlib

VAULT = pathlib.Path("/Users/henrylu/obsidian-vault")
SITE = pathlib.Path(__file__).resolve().parent.parent / "papers"

# 研究策略段落(h1 標題比對)——一律剝除
STRIP = re.compile(
    r"對你研究|為什麼讀這篇|為什麼讀\(|For My Work|可借鏡|還沒搞懂|想驗證"
    r"|跨領域對照|邊緣端應用探討|與 RGB-IR 整合探討"
)

PAPERS = {
    "vlacfdet.md": "Papers/00_主論文/VL-ACFDet (IEEE MM 2025).md",
    "yoloe.md": "Papers/00_主論文/YOLOE (ICCV 2025).md",
    "yoloworld.md": "Papers/00_主論文/YOLO-World (CVPR 2024).md",
    "rtdetr.md": "Papers/00_主論文/RT-DETR (CVPR 2024).md",
    "rfdetr.md": "Papers/00_主論文/RF-DETR (ICLR 2026).md",
    "eventgpt.md": "Papers/00_主論文/EventGPT (CVPR 2025).md",
}

RETITLE = {
    "eventgpt.md": [("# 背景:event camera 是什麼?為何對你重要?", "# 背景:event camera 是什麼?")],
}

# 歷次手術補丁:(舊, 新)。找不到舊字串時警告——通常代表 vault 原文改了,要人工複核。
PATCHES = {
    "eventgpt.md": [
        (" —— 對你而言,**event 是繼 IR 之後第三個「RGB 補充模態」,且天生最適合邊緣**(稀疏、微秒延遲、高動態範圍、低功耗)。",
         "。event 模態天生適合邊緣裝置:稀疏、微秒延遲、高動態範圍、低功耗。"),
        ("> **對你的關鍵**:event 的強項(低光、高速、HDR、低功耗)正好補 RGB 的弱項 —— 跟 IR 一樣是「退化場景的救援模態」,但 event 額外贏在**時間解析度 + 邊緣功耗**。這就是把它拉進 RGB-IR 多光譜的理由(§9)。",
         "> **關鍵視角**:event 的強項(低光、高速、HDR、低功耗)正好補 RGB 的弱項 —— 跟 IR 一樣是「退化場景的救援模態」,但 event 額外贏在**時間解析度 + 邊緣功耗**。"),
    ],
    "rfdetr.md": [
        ("⚠️ 它**不是 OVD、是 specialist** —— 對你這個做 OVD 的人,是必須認真對待的對立觀點。",
         "⚠️ 它**不是 OVD、是 specialist** —— 對 OVD 路線而言,是一個必須認真對待的對立觀點。"),
        ("> ⚠️ **對你的直接意涵**:你做「邊緣多光譜 **OVD**」。RF-DETR 逼你回答:**你的巡檢是「開放詞彙(詞常變/要 zero-shot 新類)」還是「固定類別」?** 若固定 → specialist + NAS 可能更準更省。建議在 🧭 研究架構_邊緣多光譜 OVD 明確論證「為什麼我的任務值得 OVD」。",
         "> ⚠️ **設計啟示**:RF-DETR 把「開放詞彙 vs 固定類別」變成每個部署案都得先回答的問題——若類別固定,specialist + NAS 可能更準更省;若詞彙常變、需要 zero-shot 新類,才值得付 OVD 的成本。"),
        ("**一次訓練、搜出數千種配置的 accuracy-latency Pareto 曲線(不重訓)**。",
         "**一次訓練、搜出數千種配置的 accuracy-latency Pareto 曲線**(不重訓)。"),
        ("**五個 tunable knobs(Fig 3)**——", "**五個 tunable knobs**(Fig 3)——"),
        ("**RF-DETR-2XL 勝 GroundingDINO(tiny) 與 LLMDet (CVPR 2025)(tiny)**,",
         "**RF-DETR-2XL 勝 GroundingDINO(tiny) 與 LLMDet(tiny)**(CVPR 2025),"),
        ("**NAS 一次打版試穿數千種版型(不用每件重做)**,", "**NAS 一次打版試穿數千種版型**(不用每件重做),"),
        ("「**Are Specialist Detectors Over-Optimized for COCO?**」",
         "**「Are Specialist Detectors Over-Optimized for COCO?」**"),
        ("拿一件**大廠半成品(DINOv2 預訓)**、", "拿一件**大廠半成品**(DINOv2 預訓)、"),
    ],
    "rtdetr.md": [
        ("(雲端 / 邊緣 / 你的 6GB 卡)", "(雲端 / 邊緣裝置)"),
    ],
    "yoloe.md": [
        ("(Table 1)。對你:**它是 Bridging 現用 YOLO-World head 的直接升級,你邊緣判別式即時架構的 head 首選**,且 prompt-free 免 LLM。",
         "(Table 1)。可視為 YOLO-World head 的直接升級,且 prompt-free 模式不依賴 LLM。"),
    ],
    "yoloworld.md": [
        ("**Bridging 直接用它的 detection head;它也是 YOLOE (ICCV 2025) 的前身、你即時頭三選一的基準**。",
         "**Bridging 直接用它的 detection head;它也是 YOLOE (ICCV 2025) 的前身**,常被當作即時 OVD head 的比較基準。"),
    ],
    "vlacfdet.md": [
        ("| **自建(教授實驗車)** |", "| **自建資料集(論文團隊車載擷取)** |"),
        ("、**+4.78%**", "、 **+4.78%**"),
        ("+ FPS↑(47→52)**——", "+ FPS↑(47→52)** ——"),
    ],
}

# 隱私掃描:命中即失敗,除非該行含豁免片段(教學用泛稱「你」)
LEAK_KEYWORDS = ["你的", "對你", "扣你", "你研究", "你必須", "你就站", "6GB",
                 "本筆記由", "研究定位", "北極星", "教授", "實驗室", "前身",
                 "研究架構", "實驗計畫", "巡檢"]
ALLOW_SNIPPETS = [
    "均碼(你的體型",          # rfdetr 西裝比喻(教學用泛稱)
    "按你的體型改",            # 同上
    "只合你穿",                # 同上
    "常被當作即時 OVD head",   # yoloworld:公開事實(YOLO-World 為 YOLOE 前身)
    "為你的資料集訂做",         # rfdetr:論文自己的口號(泛稱)
]

def strip_frontmatter(t):
    if t.startswith("---"):
        end = t.find("\n---", 3)
        return t[end + 4:].lstrip("\n")
    return t

def transform(src, out):
    t = (VAULT / src).read_text()
    t = strip_frontmatter(t)
    for old, new in RETITLE.get(out, []):
        t = t.replace(old, new)
    i = t.find("\n# ")
    if not t.startswith("# ") and i > 0:
        t = t[i + 1:]                      # 掉第一個 h1 前的導讀引言
    kept = []
    for b in re.split(r"(?m)(?=^# )", t):
        m = re.match(r"^# (.+)", b)
        if not (m and STRIP.search(m.group(1))):
            kept.append(b)
    t = "".join(kept)
    t = re.sub(r"(?m)^!\[\[[^\]]+\]\]\s*$", "", t)   # 圖片嵌入整行移除(站上無圖檔)
    for tag, emo in [("tip", "💡"), ("warning", "⚠️"), ("abstract", "📋"), ("success", "✅"),
                     ("important", "❗"), ("info", "ℹ️"), ("note", "📝"), ("question", "❓"), ("quote", "💬")]:
        t = re.sub(rf"> \[!{tag}\][+-]?", f"> {emo}", t)
    t = re.sub(r"\[\[[^\]|#]*#([^\]|]+)\|([^\]]+)\]\]", r"\2", t)
    t = re.sub(r"\[\[[^\]|#]*#([^\]|]+)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]|#]+)\|([^\]]+)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]|#]+)\]\]", r"\1", t)
    t = re.sub(r"\*\*\(([a-d])\)\*\*", r"(\1)", t)   # 枚舉標記去粗體(CJK 句讀後 marked 解析失敗)
    for old, new in PATCHES.get(out, []):
        if old in t:
            t = t.replace(old, new, 1)
        else:
            print(f"⚠️ {out}: 補丁失配(原文可能已改),請人工複核 → {old[:40]}…")
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    (SITE / out).write_text(t)

def lint():
    import subprocess
    ok = True
    node_lint = pathlib.Path(__file__).parent / "lint-emphasis.mjs"
    r = subprocess.run(["node", str(node_lint)], capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0:
        ok = False
    for f in sorted(SITE.glob("*.md")):
        t = f.read_text()
        if re.search(r"(?m)^!\S+\.(png|jpg|jpeg|gif)", t):
            print(f"⚠️ {f.name}: 殘留圖片嵌入文字")
            ok = False
        lines = t.split("\n")
        for k in LEAK_KEYWORDS:
            for ln, line in enumerate(lines, 1):
                if k in line and not any(s in line for s in ALLOW_SNIPPETS):
                    print(f"✗ {f.name}:L{ln} 隱私關鍵詞「{k}」: {line.strip()[:70]}")
                    ok = False
    return ok

if __name__ == "__main__":
    for out, src in PAPERS.items():
        transform(src, out)
        print(f"✓ {out}")
    sys.exit(0 if lint() else 1)
