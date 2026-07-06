/* ==========================================================================
   demo-yoloworld.js — 「三種偵測範式:誰把詞彙表的成本付在哪?」互動 demo(自包含模組)
   用法:
     <div id="demo-yoloworld"></div>
     <script src="papers/demo-yoloworld.js"></script>
     <script>initYoloworldDemo(document.getElementById('demo-yoloworld'))</script>
   規格:
     - window.initYoloworldDemo(container):在 container 內建立整個 demo;重複 init 先清舊
     - 樣式僅注入一次(#ywd-demo-style),class 全部 ywd- 前綴,不污染全域
     - 無外部資源;支援 ~350px 窄容器;prefers-reduced-motion 直接呈現最終狀態
   數據來源:papers/yoloworld.md(§2 三種偵測範式對比、§4 Table 2 LVIS zero-shot);
   堆疊條時間比例與詞彙表範例為教學示意,非實測 profiling
   ========================================================================== */
(function () {
  'use strict';

  var STYLE_ID = 'ywd-demo-style';
  var IMAGES = 10;          // 模擬張數
  var MS_PER_UNIT = 0.7;    // 動畫速度:1 示意單位 = 0.7ms(線上欄總長約 3.2 秒)

  /* ---------- 耗時模型(相對示意單位 ×100,整數避免浮點誤差) ----------
     vis=視覺 backbone、txt=文字 encode、fuse=融合、pre=一次性離線 encode
     比例為概念示意(非實測):只表達「文字成本付在哪」——
     線上 OVD 每張都付 txt;YOLO-World 只在開頭付一次 pre,融合已 re-param 折入(近零) */
  var COST = {
    trad:   { pre: 0,   per: [['vis', 100]] },
    online: { pre: 0,   per: [['vis', 160], ['txt', 220], ['fuse', 80]] },
    yw:     { pre: 200, per: [['vis', 100], ['fuse', 15]] }
  };

  /* ---------- LVIS zero-shot 數字(yoloworld.md §4,原論文 Table 2;
       Fixed AP @ minival、V100;YOLO-World 括號=原版含 RepVL-PAN,括號外=re-param 部署版) */
  var ROWS = [
    { n: 'GLIP-T',              p: '232M',       f: '0.12',       ap: 26.0 },
    { n: 'Grounding DINO-T',    p: '172M',       f: '1.5',        ap: 27.4 },
    { n: 'DetCLIP-T',           p: '155M',       f: '2.3',        ap: 34.4 },
    { n: 'YOLO-World-S',        p: '13M(77M)',   f: '74.1(19.9)', ap: 26.2, hi: 1 },
    { n: 'YOLO-World-L',        p: '48M(110M)',  f: '52.0(17.6)', ap: 35.0, hi: 1 },
    { n: 'YOLO-World-L(+CC3M)', p: '48M',        f: '52.0',       ap: 35.4, hi: 2 }
  ];
  var AP_MAX = 40; // AP 條滿格刻度

  var VOCABS = [['person', 'car'], ['dog', 'frisbee', 'backpack']]; // 詞彙表範例(示意)

  /* ---------- 樣式(注入一次) ---------- */
  var CSS = `
.ywd-root{
  --ywd-ink:#1f2328;--ywd-muted:#57606a;--ywd-line:#d0d7de;--ywd-accent:#0969da;--ywd-bg:#f6f8fa;
  --ywd-orange:#ec7500;--ywd-yellow:#e0ac00;--ywd-teal:#00bfbc;--ywd-green:#08b94e;--ywd-red:#cf222e;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;
  color:var(--ywd-ink);background:#fff;border:1px solid var(--ywd-line);border-radius:16px;
  box-shadow:0 8px 24px rgba(0,0,0,.06);padding:18px 18px 20px;line-height:1.65;
}
.ywd-title{margin:0 0 2px;font-size:1.02rem;font-weight:800}
.ywd-guide{margin:0 0 12px;font-size:.84rem;color:var(--ywd-muted)}
/* ---- 三欄範式卡片 ---- */
.ywd-cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:4px 0 14px}
.ywd-card{border:1px solid var(--ywd-line);border-top:3px solid var(--ywd-line);border-radius:12px;
  background:#fff;padding:10px 10px 12px;display:flex;flex-direction:column;gap:7px;min-width:0}
.ywd-card--trad{border-top-color:#8b949e}
.ywd-card--online{border-top-color:var(--ywd-orange)}
.ywd-card--yw{border-top-color:var(--ywd-accent)}
.ywd-card-h{font-size:.9rem;font-weight:800;line-height:1.45;margin:0}
.ywd-tag{display:inline-block;font-size:.68rem;font-weight:700;color:var(--ywd-muted);
  background:var(--ywd-bg);border:1px solid var(--ywd-line);border-radius:999px;padding:0 7px;
  margin-left:4px;vertical-align:1px;white-space:nowrap}
.ywd-svg{display:block;width:100%;height:auto}
.ywd-svg text{font-family:inherit;user-select:none;-webkit-user-select:none;pointer-events:none}
.ywd-feat{font-size:.78rem;color:var(--ywd-muted);margin:0}
.ywd-feat b{color:var(--ywd-ink)}
/* SVG 元件 */
.ywd-bx{stroke-width:1.4}
.ywd-bx--plain{fill:#f6f8fa;stroke:#8b949e}
.ywd-bx--blue{fill:#ddf4ff;stroke:var(--ywd-accent)}
.ywd-bx--blueDeep{fill:#ddf4ff;stroke:var(--ywd-accent);stroke-width:2}
.ywd-bx--orange{fill:#fff1e5;stroke:var(--ywd-orange)}
.ywd-bx--orangeHot{fill:#fff1e5;stroke:var(--ywd-orange);stroke-width:2.2}
.ywd-bx--yellow{fill:#fff8c5;stroke:var(--ywd-yellow)}
.ywd-bx--red{fill:#ffebe9;stroke:var(--ywd-red)}
.ywd-bx--teal{fill:#d9f8f7;stroke:var(--ywd-teal)}
.ywd-bx--green{fill:#dafbe1;stroke:var(--ywd-green)}
.ywd-bt{font-weight:700;fill:var(--ywd-ink);text-anchor:middle}
.ywd-sn{font-size:9.5px;font-weight:700;fill:var(--ywd-muted);text-anchor:middle}
.ywd-sn2{font-size:8.5px;fill:var(--ywd-muted)}
.ywd-zl{font-size:8.8px;font-weight:800;fill:#9a6700}
.ywd-ln{stroke:#8b949e;stroke-width:1.6;fill:none}
.ywd-ln--dash{stroke-dasharray:4 3}
.ywd-ah{fill:#8b949e}
.ywd-zone{fill:#fff8c5;fill-opacity:.4;stroke:var(--ywd-yellow);stroke-width:1.4;stroke-dasharray:5 4}
/* 詞彙表變更後的反應徽章 */
.ywd-badge{margin-top:auto;font-size:.75rem;font-weight:700;border-radius:9px;padding:6px 9px;
  line-height:1.5;opacity:0;transform:translateY(4px);transition:opacity .35s ease,transform .35s ease}
.ywd-badge.ywd-show{opacity:1;transform:none}
.ywd-badge--err{background:#ffebe9;border:1px solid #ffb3ab;color:var(--ywd-red);animation:ywd-shake .5s ease}
.ywd-badge--warn{background:#fff8c5;border:1px solid #eed888;color:#7d4e00}
.ywd-badge--run{background:#ddf4ff;border:1px solid #a3d3ff;color:#0550ae}
.ywd-badge--ok{background:#dafbe1;border:1px solid #a5e8b8;color:#116329}
@keyframes ywd-shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-4px)}40%{transform:translateX(4px)}60%{transform:translateX(-3px)}80%{transform:translateX(3px)}}
.ywd-reenc{display:block;height:7px;margin-top:5px;background:#fff;border:1px solid var(--ywd-yellow);border-radius:4px;overflow:hidden}
.ywd-reenc i{display:block;height:100%;width:0;background:repeating-linear-gradient(45deg,var(--ywd-yellow) 0 4px,#f7d84b 4px 8px);transition:width .8s ease}
/* ---- 詞彙表 + 控制列 ---- */
.ywd-vocab{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:0 0 10px}
.ywd-vlab{font-size:.78rem;font-weight:700;color:var(--ywd-muted)}
.ywd-chip{font-size:.8rem;font-weight:700;color:var(--ywd-accent);background:#ddf4ff;
  border:1px solid var(--ywd-accent);border-radius:999px;padding:1px 10px}
.ywd-chip--pop{animation:ywd-pop .45s cubic-bezier(.34,1.56,.64,1)}
@keyframes ywd-pop{from{transform:scale(.55);opacity:0}}
.ywd-controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px}
.ywd-btn{font:inherit;font-size:.86rem;font-weight:700;padding:7px 14px;border-radius:9px;cursor:pointer;
  border:1px solid var(--ywd-accent);background:var(--ywd-accent);color:#fff;transition:filter .2s,opacity .2s}
.ywd-btn:hover{filter:brightness(1.1)}
.ywd-btn:disabled{opacity:.5;cursor:default;filter:none}
.ywd-btn--ghost{background:#fff;color:var(--ywd-accent)}
/* ---- 模擬區 ---- */
.ywd-sim{border:1px solid var(--ywd-line);border-radius:12px;background:var(--ywd-bg);padding:12px 12px 6px;margin:0 0 16px}
.ywd-sec-t{font-size:.88rem;font-weight:800;margin:0 0 8px}
.ywd-legend{display:flex;gap:10px 14px;flex-wrap:wrap;font-size:.72rem;color:var(--ywd-muted);margin:0 0 10px}
.ywd-lg{display:inline-flex;align-items:center;gap:5px}
.ywd-sw{width:10px;height:10px;border-radius:3px;flex:none}
.ywd-sw--vis{background:var(--ywd-accent)}
.ywd-sw--txt{background:var(--ywd-orange)}
.ywd-sw--fuse{background:var(--ywd-teal)}
.ywd-sw--off{background:repeating-linear-gradient(45deg,var(--ywd-yellow) 0 4px,#f7d84b 4px 8px)}
.ywd-lane{margin:0 0 10px}
.ywd-lane-h{display:flex;align-items:baseline;gap:8px;font-size:.78rem;margin:0 0 4px}
.ywd-lane-name{font-weight:800;white-space:nowrap}
.ywd-lane-count{color:var(--ywd-muted);font-variant-numeric:tabular-nums}
.ywd-lane-total{margin-left:auto;font-weight:800;color:var(--ywd-accent);opacity:0;transition:opacity .3s;white-space:nowrap}
.ywd-lane--done .ywd-lane-total{opacity:1}
.ywd-track{display:flex;height:16px;background:#fff;border:1px solid var(--ywd-line);border-radius:8px;overflow:hidden}
.ywd-seg{height:100%;width:0;flex:none}
.ywd-seg--vis{background:var(--ywd-accent)}
.ywd-seg--txt{background:var(--ywd-orange)}
.ywd-seg--fuse{background:var(--ywd-teal)}
.ywd-seg--off{background:repeating-linear-gradient(45deg,var(--ywd-yellow) 0 4px,#f7d84b 4px 8px)}
.ywd-seg--tick{box-shadow:inset -1px 0 0 rgba(255,255,255,.9)}
.ywd-result{font-size:.79rem;background:#fff;border:1px solid var(--ywd-line);border-radius:10px;padding:9px 11px;margin:2px 0 10px}
.ywd-result b{color:var(--ywd-accent)}
/* ---- 數字卡 ---- */
.ywd-stats{margin:0 0 4px}
.ywd-twrap{border:1px solid var(--ywd-line);border-radius:10px;overflow-x:auto;-webkit-overflow-scrolling:touch;background:#fff}
.ywd-table{width:100%;min-width:320px;border-collapse:collapse;font-size:.76rem}
.ywd-table th,.ywd-table td{padding:5px 9px;border-bottom:1px solid var(--ywd-line);text-align:left;white-space:nowrap}
.ywd-table tr:last-child td{border-bottom:none}
.ywd-table th{background:var(--ywd-bg);font-size:.7rem;color:var(--ywd-muted)}
.ywd-table td.ywd-num,.ywd-table th.ywd-num{text-align:right;font-variant-numeric:tabular-nums}
.ywd-hi td{background:rgba(221,244,255,.45)}
.ywd-hi2 td{background:#ddf4ff;font-weight:800}
.ywd-apbar{display:inline-block;width:52px;height:7px;background:var(--ywd-bg);border:1px solid var(--ywd-line);border-radius:4px;margin-right:6px;vertical-align:1px}
.ywd-apbar i{display:block;height:100%;background:var(--ywd-accent);border-radius:3px}
.ywd-callout{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0 0}
.ywd-co{font-size:.72rem;font-weight:700;background:var(--ywd-bg);border:1px solid var(--ywd-line);
  border-left:3px solid var(--ywd-accent);border-radius:8px;padding:4px 9px}
.ywd-note{font-size:.72rem;color:var(--ywd-muted);margin:12px 0 0;line-height:1.6}
/* ---- 響應式 ---- */
@media (max-width:720px){
  .ywd-cols{grid-template-columns:1fr}
}
@media (max-width:420px){
  .ywd-root{padding:14px 12px 16px}
  .ywd-btn{font-size:.82rem;padding:6px 11px}
  .ywd-table{font-size:.7rem}
}
@media (prefers-reduced-motion:reduce){
  .ywd-root *,.ywd-root *::before,.ywd-root *::after{transition-duration:.01ms!important;transition-delay:0ms!important;animation-duration:.01ms!important;animation-delay:0ms!important}
}
`;

  /* ---------- SVG 小工具 ---------- */
  function bx(x, y, w, h, tone, lines, fs) {
    var cx = x + w / 2;
    var size = fs || (lines.length > 1 ? 8.6 : 9.6);
    var lh = size + 2.6;
    var y0 = y + h / 2 - (lines.length - 1) * lh / 2 + size * 0.36;
    var out = '<rect class="ywd-bx ywd-bx--' + tone + '" x="' + x + '" y="' + y +
      '" width="' + w + '" height="' + h + '" rx="6"/>';
    for (var i = 0; i < lines.length; i++) {
      out += '<text class="ywd-bt" style="font-size:' + size + 'px" x="' + cx +
        '" y="' + (y0 + i * lh) + '">' + lines[i] + '</text>';
    }
    return out;
  }
  function aV(x, y1, y2) { // 垂直箭頭(向下)
    return '<line class="ywd-ln" x1="' + x + '" y1="' + y1 + '" x2="' + x + '" y2="' + (y2 - 4) + '"/>' +
      '<path class="ywd-ah" d="M' + (x - 3.6) + ' ' + (y2 - 4.6) + ' L' + (x + 3.6) + ' ' + (y2 - 4.6) + ' L' + x + ' ' + y2 + ' Z"/>';
  }
  function aH(x1, x2, y) { // 水平箭頭(向右)
    return '<line class="ywd-ln" x1="' + x1 + '" y1="' + y + '" x2="' + (x2 - 4) + '" y2="' + y + '"/>' +
      '<path class="ywd-ah" d="M' + (x2 - 4.6) + ' ' + (y - 3.6) + ' L' + (x2 - 4.6) + ' ' + (y + 3.6) + ' L' + x2 + ' ' + y + ' Z"/>';
  }
  function pline(pts) {
    return '<polyline class="ywd-ln" points="' + pts + '"/>';
  }

  /* (a) 傳統:影像→Backbone→Head→固定 N 類 */
  function svgTrad() {
    return '<svg class="ywd-svg" viewBox="0 0 230 158" role="img" aria-label="傳統偵測器:影像經 Backbone、Head 輸出固定類別">' +
      bx(62, 6, 106, 22, 'blue', ['影像']) +
      aV(115, 28, 40) +
      bx(62, 42, 106, 22, 'plain', ['Backbone']) +
      aV(115, 64, 76) +
      bx(62, 78, 106, 22, 'plain', ['Head']) +
      aV(115, 100, 112) +
      bx(40, 114, 150, 24, 'red', ['固定 N 類(COCO 80 類)']) +
      '<text class="ywd-sn" x="115" y="153">🔒 換類別 → 要重新訓練</text>' +
      '</svg>';
  }
  /* (b) 線上 encode OVD:影像+文字每次推論都過 text encoder */
  function svgOnline() {
    return '<svg class="ywd-svg" viewBox="0 0 230 158" role="img" aria-label="線上 encode OVD:影像與文字每次推論同時編碼再融合">' +
      bx(10, 6, 96, 22, 'blue', ['影像']) +
      bx(124, 6, 96, 22, 'orange', ['文字提示']) +
      aV(58, 28, 40) + aV(172, 28, 40) +
      bx(10, 42, 96, 26, 'plain', ['大 detector']) +
      bx(124, 42, 96, 26, 'orangeHot', ['Text Encoder', '⟳ 每次推論都跑'], 8.4) +
      pline('58,68 58,82 115,82') + pline('172,68 172,82 115,82') +
      aV(115, 82, 96) +
      bx(70, 98, 90, 22, 'teal', ['融合(跨模態)']) +
      aV(115, 120, 132) +
      bx(70, 134, 90, 22, 'plain', ['偵測結果']) +
      '</svg>';
  }
  /* (c) YOLO-World:離線 encode 一次 → re-param 進權重 → 推論即時 */
  function svgYW() {
    return '<svg class="ywd-svg" viewBox="0 0 230 158" role="img" aria-label="YOLO-World:提示先離線編成 offline vocabulary 再 re-parameterize 進權重,推論即時">' +
      '<rect class="ywd-zone" x="4" y="4" width="222" height="64" rx="8"/>' +
      '<text class="ywd-zl" x="12" y="17">🗄 離線(一次付清)— offline vocabulary</text>' +
      bx(10, 30, 50, 24, 'plain', ['提示詞']) +
      aH(60, 70, 42) +
      bx(70, 26, 94, 32, 'yellow', ['CLIP Text Encoder', '(凍結)'], 8.4) +
      aH(164, 174, 42) +
      bx(174, 30, 48, 24, 'yellow', ['embedding', '檔'], 7.8) +
      '<line class="ywd-ln ywd-ln--dash" x1="115" y1="68" x2="115" y2="82"/>' +
      '<path class="ywd-ah" d="M111.4 81.4 L118.6 81.4 L115 86 Z"/>' +
      '<text class="ywd-sn2" x="122" y="80">re-param 進權重</text>' +
      bx(8, 92, 46, 26, 'blue', ['影像']) +
      aH(54, 62, 105) +
      bx(62, 86, 106, 38, 'blueDeep', ['YOLOv8 backbone', '+ RepVL-PAN', '(融合已折入權重)'], 8.2) +
      aH(168, 176, 105) +
      bx(176, 92, 46, 26, 'green', ['⚡ 即時', '偵測'], 8.6) +
      '<text class="ywd-sn" x="115" y="150">🚀 換詞彙=換一份 embedding,不重訓</text>' +
      '</svg>';
  }

  /* ---------- 文案(機制描述取自 yoloworld.md §2 範式對比) ---------- */
  var PARADIGMS = [
    {
      key: 'trad', cls: 'trad', emoji: '🔒', name: '傳統偵測器', tag: '(a) 固定詞彙',
      svg: svgTrad,
      feat: '影像 → Backbone → Head → 固定 N 類(如 COCO 80 類)。沒有文字分支、負擔最小,但<b>換類別要重新訓練</b>——詞彙受限。'
    },
    {
      key: 'online', cls: 'online', emoji: '🐢', name: '線上 encode OVD', tag: '(b) GLIP / G-DINO 式',
      svg: svgOnline,
      feat: '影像+文字<b>每次推論都過 text encoder</b>(online vocabulary,文字影像同時編碼)。詞彙隨時換,但大 detector + 每張都付文字成本 → <b>慢、重</b>。'
    },
    {
      key: 'yw', cls: 'yw', emoji: '⚡', name: 'prompt-then-detect', tag: '(c) YOLO-World',
      svg: svgYW,
      feat: '詞彙表<b>離線先 encode 成 offline vocabulary(embedding)存下來</b>,再 re-parameterize 進權重;推論只跑視覺+已折疊的輕量融合 → <b>即時、換詞彙不重訓</b>,部署零額外成本。'
    }
  ];

  var LANES = [
    { key: 'trad', label: '🔒 傳統' },
    { key: 'online', label: '🐢 線上 encode' },
    { key: 'yw', label: '⚡ YOLO-World' }
  ];

  /* ---------- 耗時展開 ---------- */
  function laneSegs(key) {
    var c = COST[key], segs = [], imgEnds = [], t = 0, k, i, seg;
    if (c.pre) { segs.push({ type: 'off', start: 0, end: c.pre, tick: true }); t = c.pre; }
    for (k = 1; k <= IMAGES; k++) {
      for (i = 0; i < c.per.length; i++) {
        seg = { type: c.per[i][0], start: t, end: t + c.per[i][1], tick: i === c.per.length - 1 };
        segs.push(seg); t = seg.end;
        if (seg.tick) imgEnds.push(t);
      }
    }
    return { segs: segs, imgEnds: imgEnds, total: t };
  }
  var PLAN = {};
  var MAXT = 0;
  LANES.forEach(function (l) { PLAN[l.key] = laneSegs(l.key); if (PLAN[l.key].total > MAXT) MAXT = PLAN[l.key].total; });

  function fmtU(u) { return (u / 100).toFixed(1).replace(/\.0$/, ''); }

  /* ---------- HTML ---------- */
  function buildHTML() {
    var h = '<div class="ywd-root">';
    h += '<p class="ywd-title">🎮 互動走讀|prompt-then-detect:把文字成本搬到離線</p>';
    h += '<p class="ywd-guide">三種偵測範式:誰把詞彙表的成本付在哪?(機制取自 yoloworld.md §2 範式對比,數字取自 §4 實驗)</p>';

    /* 三欄卡片 */
    h += '<div class="ywd-cols">';
    PARADIGMS.forEach(function (p) {
      h += '<section class="ywd-card ywd-card--' + p.cls + '">' +
        '<h4 class="ywd-card-h">' + p.emoji + ' ' + p.name + '<span class="ywd-tag">' + p.tag + '</span></h4>' +
        p.svg() +
        '<p class="ywd-feat">特點:' + p.feat + '</p>' +
        '<div class="ywd-badge" data-badge="' + p.key + '" hidden></div>' +
        '</section>';
    });
    h += '</div>';

    /* 詞彙表 + 控制 */
    h += '<div class="ywd-vocab"><span class="ywd-vlab">目前詞彙表:</span><span class="ywd-chips"></span></div>';
    h += '<div class="ywd-controls">' +
      '<button type="button" class="ywd-btn ywd-run">▶ 模擬跑 10 張影像</button>' +
      '<button type="button" class="ywd-btn ywd-btn--ghost ywd-swap">🔁 更換詞彙表</button>' +
      '</div>';

    /* 模擬堆疊條 */
    h += '<div class="ywd-sim"><p class="ywd-sec-t">⏱ 核心互動:同一批 10 張影像,時間花在哪?(相對示意)</p>';
    h += '<div class="ywd-legend">' +
      '<span class="ywd-lg"><i class="ywd-sw ywd-sw--vis"></i>視覺 backbone</span>' +
      '<span class="ywd-lg"><i class="ywd-sw ywd-sw--txt"></i>文字 encode(每張)</span>' +
      '<span class="ywd-lg"><i class="ywd-sw ywd-sw--fuse"></i>融合</span>' +
      '<span class="ywd-lg"><i class="ywd-sw ywd-sw--off"></i>離線 encode(一次性)</span>' +
      '</div>';
    LANES.forEach(function (l) {
      h += '<div class="ywd-lane" data-lane="' + l.key + '">' +
        '<div class="ywd-lane-h"><span class="ywd-lane-name">' + l.label + '</span>' +
        '<span class="ywd-lane-count">0/' + IMAGES + ' 張</span>' +
        '<span class="ywd-lane-total"></span></div>' +
        '<div class="ywd-track" aria-hidden="true"></div>' +
        '</div>';
    });
    h += '<div class="ywd-result" hidden></div></div>';

    /* 數字卡(Table 2) */
    h += '<div class="ywd-stats"><p class="ywd-sec-t">📊 數字驗證:LVIS zero-shot(yoloworld.md §4,原論文 Table 2;minival、V100)</p>';
    h += '<div class="ywd-twrap"><table class="ywd-table"><thead><tr>' +
      '<th>模型</th><th class="ywd-num">Params</th><th class="ywd-num">FPS</th><th>AP(Fixed)</th>' +
      '</tr></thead><tbody>';
    ROWS.forEach(function (r) {
      var cls = r.hi === 2 ? ' class="ywd-hi2"' : (r.hi === 1 ? ' class="ywd-hi"' : '');
      h += '<tr' + cls + '><td>' + r.n + '</td>' +
        '<td class="ywd-num">' + r.p + '</td>' +
        '<td class="ywd-num">' + r.f + '</td>' +
        '<td><span class="ywd-apbar"><i style="width:' + (r.ap / AP_MAX * 100).toFixed(1) + '%"></i></span>' + r.ap.toFixed(1) + '</td></tr>';
    });
    h += '</tbody></table></div>';
    h += '<div class="ywd-callout">' +
      '<span class="ywd-co">⚡ 比 Grounding DINO-T:+8 AP、快 35×</span>' +
      '<span class="ywd-co">🏆 35.4 AP 超越 DetCLIP-T,推論快 ~20×</span>' +
      '<span class="ywd-co">🔧 re-param(S 版):77M/19.9 FPS → 13M/74.1 FPS(參數少 6×、快 3.7×)</span>' +
      '</div></div>';

    /* 註記 */
    h += '<p class="ywd-note">⚠️ 堆疊條的時間比例為<b>概念示意</b>(非實測 profiling),只表達「文字成本付在哪」;詞彙表範例亦為示意。' +
      '表格數字出自 papers/yoloworld.md §4 實驗節(原論文 Table 2:LVIS zero-shot、Fixed AP @ minival、V100;' +
      'YOLO-World 括號外=re-parameterized 部署版、括號內=原版含 RepVL-PAN)。</p>';

    h += '</div>';
    return h;
  }

  /* ---------- 樣式注入(單次) ---------- */
  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement('style');
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  /* ---------- 入口 ---------- */
  window.initYoloworldDemo = function (container) {
    if (typeof container === 'string') container = document.querySelector(container);
    if (!container) return;
    if (typeof container.__ywdCleanup === 'function') container.__ywdCleanup(); // 重複 init 先清舊
    ensureStyle();
    container.innerHTML = buildHTML();

    var REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    var root = container.querySelector('.ywd-root');
    var runBtn = root.querySelector('.ywd-run');
    var swapBtn = root.querySelector('.ywd-swap');
    var chipsEl = root.querySelector('.ywd-chips');
    var resultEl = root.querySelector('.ywd-result');
    var laneEls = {}, badgeEls = {};
    LANES.forEach(function (l) { laneEls[l.key] = root.querySelector('.ywd-lane[data-lane="' + l.key + '"]'); });
    PARADIGMS.forEach(function (p) { badgeEls[p.key] = root.querySelector('.ywd-badge[data-badge="' + p.key + '"]'); });

    var rafId = 0, timers = [], running = false, vocabIdx = 0, segDoms = {};

    function cleanup() {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      timers.forEach(clearTimeout); timers.length = 0;
      running = false;
    }
    container.__ywdCleanup = cleanup;
    function later(fn, ms) { timers.push(setTimeout(fn, REDUCED ? 0 : ms)); }

    /* ----- 詞彙表 chips ----- */
    function renderVocab(pop) {
      chipsEl.innerHTML = '';
      VOCABS[vocabIdx].forEach(function (w, i) {
        var c = document.createElement('span');
        c.className = 'ywd-chip' + (pop ? ' ywd-chip--pop' : '');
        if (pop) c.style.animationDelay = (i * 70) + 'ms';
        c.textContent = w;
        chipsEl.appendChild(c);
      });
    }
    renderVocab(false);

    /* ----- 模擬跑 10 張 ----- */
    function buildTracks() {
      LANES.forEach(function (l) {
        var track = laneEls[l.key].querySelector('.ywd-track');
        track.innerHTML = '';
        segDoms[l.key] = PLAN[l.key].segs.map(function (s) {
          var d = document.createElement('i');
          d.className = 'ywd-seg ywd-seg--' + s.type + (s.tick ? ' ywd-seg--tick' : '');
          track.appendChild(d);
          return { el: d, start: s.start, end: s.end };
        });
        laneEls[l.key].classList.remove('ywd-lane--done');
        laneEls[l.key].querySelector('.ywd-lane-count').textContent = '0/' + IMAGES + ' 張';
        laneEls[l.key].querySelector('.ywd-lane-total').textContent = '';
      });
    }
    function setProgress(tu) {
      LANES.forEach(function (l) {
        var plan = PLAN[l.key], done = 0, i, w;
        for (i = 0; i < segDoms[l.key].length; i++) {
          var s = segDoms[l.key][i];
          w = Math.max(0, Math.min(tu, s.end) - s.start);
          s.el.style.width = (w / MAXT * 100) + '%';
        }
        for (i = 0; i < plan.imgEnds.length; i++) if (plan.imgEnds[i] <= tu) done++;
        laneEls[l.key].querySelector('.ywd-lane-count').textContent = done + '/' + IMAGES + ' 張';
        if (tu >= plan.total) {
          laneEls[l.key].classList.add('ywd-lane--done');
          laneEls[l.key].querySelector('.ywd-lane-total').textContent = '🏁 ≈' + fmtU(plan.total) + ' 單位';
        }
      });
    }
    function finishRun() {
      var tT = PLAN.trad.total, tO = PLAN.online.total, tY = PLAN.yw.total, pre = COST.yw.pre;
      resultEl.innerHTML = '🏁 <b>總時間對比(相對值,示意)</b>:🔒 傳統 ' + fmtU(tT) +
        ' ‖ 🐢 線上 encode ' + fmtU(tO) +
        ' ‖ ⚡ YOLO-World ' + fmtU(tY) + '(含開頭一次性離線 encode ' + fmtU(pre) +
        ',可在部署前先做 → 線上推論部分僅 ' + fmtU(tY - pre) + ')。' +
        '橙色段=每張都重付的文字成本,正是 md §2 說「先前 OVD 慢」的主因;' +
        'YOLO-World 把它搬到離線一次付清,<b>推論就是一台乾淨 YOLO</b>。';
      resultEl.hidden = false;
      running = false;
      runBtn.disabled = false;
      runBtn.textContent = '↻ 再跑一次(10 張)';
    }
    function run() {
      if (running) return;
      running = true;
      runBtn.disabled = true;
      resultEl.hidden = true;
      buildTracks();
      if (REDUCED) { setProgress(MAXT); finishRun(); return; }
      var t0 = performance.now();
      (function loop() {
        if (!root.isConnected) { cleanup(); return; }
        var tu = (performance.now() - t0) / MS_PER_UNIT;
        if (tu >= MAXT) { setProgress(MAXT); rafId = 0; finishRun(); return; }
        setProgress(tu);
        rafId = requestAnimationFrame(loop);
      })();
    }
    runBtn.addEventListener('click', run);

    /* ----- 更換詞彙表 ----- */
    function setBadge(key, cls, html) {
      var b = badgeEls[key];
      b.hidden = false;
      b.className = 'ywd-badge';
      void b.offsetWidth; // 重新觸發動畫
      b.className = 'ywd-badge ywd-badge--' + cls + ' ywd-show';
      b.innerHTML = html;
    }
    function swapVocab() {
      swapBtn.disabled = true;
      vocabIdx = 1 - vocabIdx;
      renderVocab(!REDUCED);
      setBadge('trad', 'err', '❌ 需要重新訓練 —— 固定詞彙,新類別不在 N 類裡');
      setBadge('online', 'warn', '⭕ 模型不用改:下次推論把新詞彙重新 encode(每次都要,照樣慢)');
      setBadge('yw', 'run', '⚡ 離線 re-encode 新詞彙…<span class="ywd-reenc"><i></i></span>');
      var fill = badgeEls.yw.querySelector('.ywd-reenc i');
      requestAnimationFrame(function () { if (fill) fill.style.width = '100%'; });
      later(function () {
        setBadge('yw', 'ok', '✅ 換好:存成新 offline vocabulary(embedding 檔)→ 不重訓,之後照常即時');
        swapBtn.disabled = false;
      }, 900);
    }
    swapBtn.addEventListener('click', swapVocab);
  };
})();
