/*!
 * demo-rtdetr.js — RT-DETR 互動走讀 demo(自包含 vanilla JS,無外部資源)
 * 介面:window.initRtdetrDemo(container)
 * 事實來源:papers/rtdetr.md;凡標「示意」者為定性 / 比例呈現,非論文實測數字。
 */
(function () {
  'use strict';

  var STYLE_ID = 'rtd-demo-style';

  /* ================= 論文事實(取自 rtdetr.md) ================= */
  // 640 輸入下各尺度 token 數:S3 = 80×80 = 6400、S4 = 40×40 = 1600、S5 = 20×20 = 400
  var SCALES = [
    { key: 's3', name: 'S3', stride: '1/8',  grid: '80×80', tokens: 6400, note: '高解析/低語義' },
    { key: 's4', name: 'S4', stride: '1/16', grid: '40×40', tokens: 1600, note: '中間層' },
    { key: 's5', name: 'S5', stride: '1/32', grid: '20×20', tokens: 400,  note: '低解析/高語義' }
  ];
  var N_ALL = SCALES.reduce(function (a, s) { return a + s.tokens; }, 0); // 8400
  var BASE_COST = N_ALL * N_ALL;   // 70,560,000 — 原始 DETR 式(全尺度 concat)基準
  var RT_MS = 1000 / 108;          // ≈9.3 ms — RT-DETR-R50:T4 TensorRT FP16 108 FPS(由 FPS 換算)
  var YOLO_MS = 1000 / 71;         // ≈14.1 ms — YOLOv8-L:~71 FPS(含 NMS)(由 FPS 換算)
  var LAT_MAX = 23;                // 延遲條刻度上限(ms,純顯示用)
  // decoder 層數:論文只給趨勢(減層→提速、AP 緩降,每層有 auxiliary head 可早退),以下為定性示意
  var DEC_LAT = [50, 60, 70, 80, 90, 100];         // 相對延遲 %(示意,線性)
  var DEC_AP  = [87, 93.5, 97, 98.6, 99.5, 100];   // 相對 AP(示意,飽和曲線)

  /* ================= 小工具 ================= */
  function fmt(n) { return Math.round(n).toLocaleString('en-US'); }
  function ratioStr(r) {
    if (r >= 100) return '' + Math.round(r);
    if (r >= 10) return r.toFixed(1);
    return r.toFixed(2);
  }
  function xL(i) { return 30 + i * 42; }                       // 曲線圖 x 座標(層 i=0..5)
  function apY(ap) { return 74 - (ap - 86) * (64 / 15); }      // 相對AP → y 座標

  /* ================= 樣式(注入一次,全部 rtd- 前綴) ================= */
  var CSS = '' +
    '.rtd-demo{--rtd-accent:var(--accent,#0969da);--rtd-red:var(--red,#e93147);--rtd-green:var(--green,#08b94e);--rtd-orange:var(--orange,#ec7500);--rtd-ink:var(--ink,#1f2328);--rtd-muted:var(--muted,#57606a);--rtd-line:var(--line,#d0d7de);--rtd-bg:var(--bg,#f6f8fa);' +
      'border:1px solid var(--rtd-line);border-radius:14px;background:linear-gradient(180deg,#fbfcfd,#fff);padding:16px 16px 18px;margin:1.4em 0;color:var(--rtd-ink);' +
      'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;font-size:.92rem;line-height:1.7}' +
    '.rtd-demo,.rtd-demo *{box-sizing:border-box}' +
    '.rtd-head{display:inline-block;font-weight:700;font-size:.95rem;color:var(--rtd-accent);background:rgba(9,105,218,.09);border-radius:12px;padding:6px 14px;margin:0 0 6px}' +
    '.rtd-sec{margin-top:16px}' +
    '.rtd-sec+.rtd-sec{border-top:1px dashed var(--rtd-line);padding-top:16px;margin-top:20px}' +
    '.rtd-sec h4{margin:0 0 .4em;font-size:1.03rem;line-height:1.5}' +
    '.rtd-desc{margin:.2em 0 .8em;font-size:.86rem;color:var(--rtd-muted)}' +
    /* 尺度卡片 */
    '.rtd-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(92px,1fr));gap:8px;margin:0 0 10px}' +
    '.rtd-card{display:block;border:1.5px solid var(--rtd-line);border-radius:10px;padding:9px 10px;background:#fff;cursor:pointer;user-select:none;-webkit-user-select:none;transition:border-color .18s,box-shadow .18s,background .18s}' +
    '.rtd-card:hover{border-color:var(--rtd-accent)}' +
    '.rtd-card.rtd-on{border-color:var(--rtd-accent);background:rgba(9,105,218,.06);box-shadow:0 2px 8px rgba(9,105,218,.12)}' +
    '.rtd-cname{display:block;font-weight:700;font-size:.95rem}' +
    '.rtd-cname small{font-weight:400;color:var(--rtd-muted)}' +
    '.rtd-cgrid{display:block;font-size:.78rem;color:var(--rtd-muted);font-variant-numeric:tabular-nums}' +
    '.rtd-cgrid b{color:var(--rtd-ink)}' +
    '.rtd-tokbar{display:block;height:5px;border-radius:999px;background:var(--rtd-bg);margin:5px 0;overflow:hidden}' +
    '.rtd-tokbar i{display:block;height:100%;border-radius:999px;background:var(--rtd-accent);opacity:.55}' +
    '.rtd-cnote{display:block;font-size:.72rem;color:var(--rtd-muted);margin-bottom:5px}' +
    '.rtd-chk{display:flex;gap:6px;align-items:flex-start;font-size:.76rem;color:var(--rtd-muted);line-height:1.45}' +
    '.rtd-chk input{accent-color:var(--rtd-accent);margin:2px 0 0;flex:none}' +
    '.rtd-concat{display:flex;gap:7px;align-items:flex-start;font-size:.82rem;margin:2px 0 10px;cursor:pointer;user-select:none;-webkit-user-select:none}' +
    '.rtd-concat input{accent-color:var(--rtd-accent);margin:4px 0 0;flex:none}' +
    /* 預設情境按鈕 */
    '.rtd-presets{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}' +
    '.rtd-btn{appearance:none;font-family:inherit;font-size:.82rem;border:1.5px solid var(--rtd-line);background:#fff;color:var(--rtd-ink);padding:6px 13px;border-radius:999px;cursor:pointer;transition:all .18s ease}' +
    '.rtd-btn:hover{border-color:var(--rtd-accent);color:var(--rtd-accent);transform:translateY(-1px)}' +
    '.rtd-btn.rtd-btn-on{background:var(--rtd-accent);border-color:var(--rtd-accent);color:#fff;font-weight:600}' +
    /* 長條圖 */
    '.rtd-barrow{display:grid;grid-template-columns:88px 1fr;gap:10px;align-items:center;margin:7px 0}' +
    '.rtd-blabel{font-size:.78rem;color:var(--rtd-muted);text-align:right;line-height:1.35}' +
    '.rtd-track{position:relative;height:22px;background:var(--rtd-bg);border-radius:999px;overflow:hidden}' +
    '.rtd-track i{display:block}' +
    '.rtd-fill{position:absolute;top:0;bottom:0;left:0;width:0;border-radius:999px;transition:width .65s cubic-bezier(.22,1,.36,1),background-color .4s}' +
    '.rtd-fill.rtd-nz{min-width:4px}' +
    '.rtd-fill-base{background:var(--rtd-red);opacity:.45}' +
    '.rtd-fill-yolo{background:var(--rtd-red);transition-duration:.55s}' +
    '.rtd-fill-rt{background:var(--rtd-green)}' +
    '.rtd-fill-dec{background:var(--rtd-accent)}' +
    '.rtd-c-red{background:var(--rtd-red)}.rtd-c-orange{background:var(--rtd-orange)}.rtd-c-green{background:var(--rtd-green)}' +
    '.rtd-band{position:absolute;top:0;bottom:0;border-radius:999px;background:rgba(233,49,71,.16);transition:left .5s,width .5s}' +
    '.rtd-bval{position:absolute;right:9px;top:50%;transform:translateY(-50%);font-style:normal;font-size:.72rem;color:var(--rtd-ink);font-variant-numeric:tabular-nums;white-space:nowrap;text-shadow:0 0 4px #fff,0 0 4px #fff,0 0 6px #fff}' +
    /* 倍率與公式 */
    '.rtd-ratio{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:10px 0 2px}' +
    '.rtd-ratio-num{display:inline-block;font-size:1.5rem;line-height:1.2;font-variant-numeric:tabular-nums}' +
    '.rtd-r-green{color:var(--rtd-green)}.rtd-r-orange{color:var(--rtd-orange)}.rtd-r-red{color:var(--rtd-red)}' +
    '.rtd-ratio-cap{font-size:.78rem;color:var(--rtd-muted)}' +
    '@keyframes rtdPulse{0%{transform:scale(1)}35%{transform:scale(1.16)}100%{transform:scale(1)}}' +
    '.rtd-pulse{animation:rtdPulse .45s ease}' +
    '.rtd-formula{font-size:.8rem;color:var(--rtd-muted);font-variant-numeric:tabular-nums;background:var(--rtd-bg);border-radius:8px;padding:6px 10px;margin:8px 0;overflow-x:auto;white-space:nowrap}' +
    /* 提示與註記 */
    '.rtd-hint{font-size:.83rem;margin:8px 0;padding:8px 12px;border-radius:0 8px 8px 0;border-left:3px solid var(--rtd-accent);background:rgba(9,105,218,.05)}' +
    '.rtd-hint-good{border-left-color:var(--rtd-green);background:rgba(8,185,78,.07)}' +
    '.rtd-hint-warn{border-left-color:var(--rtd-red);background:rgba(233,49,71,.06)}' +
    '.rtd-soul{font-size:.84rem;margin:8px 0}' +
    '.rtd-annot{font-size:.75rem;color:var(--rtd-muted);margin:4px 0 0}' +
    '.rtd-cap{font-size:.76rem;color:var(--rtd-muted);margin:2px 0 10px}' +
    '.rtd-note{font-size:.76rem;color:#6b5218;background:#fff8ec;border:1px solid #f0d9ac;border-left:3px solid var(--rtd-orange);border-radius:0 8px 8px 0;padding:7px 11px;margin:10px 0 0}' +
    /* 滑桿 */
    '.rtd-slider{display:flex;flex-wrap:wrap;align-items:center;gap:10px;font-size:.86rem;margin:8px 0}' +
    '.rtd-slider b{font-variant-numeric:tabular-nums;min-width:2.2em;text-align:right}' +
    '.rtd-slider input[type=range]{flex:1 1 140px;min-width:120px;accent-color:var(--rtd-accent);margin:0}' +
    /* 曲線圖 */
    '.rtd-curve{width:100%;max-width:360px;height:auto;display:block;margin:6px 0 2px}' +
    '.rtd-svgt{font-size:9px;fill:var(--rtd-muted)}' +
    '.rtd-axis{stroke:var(--rtd-line);stroke-width:1}' +
    '.rtd-pl{fill:none;stroke:var(--rtd-accent);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}' +
    '.rtd-dot{fill:var(--rtd-accent)}' +
    /* 動效偏好 */
    '@media (prefers-reduced-motion:reduce){.rtd-demo *,.rtd-demo *::before,.rtd-demo *::after{transition:none!important;animation:none!important}}';

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var el = document.createElement('style');
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
  }

  /* ================= HTML 模板 ================= */
  function buildHTML() {
    var cards = SCALES.map(function (s) {
      var tokPct = (s.tokens / SCALES[0].tokens * 100).toFixed(2);
      return '<label class="rtd-card" data-scale="' + s.key + '">' +
        '<span class="rtd-cname">' + s.name + ' <small>(' + s.stride + ')</small></span>' +
        '<span class="rtd-cgrid">' + s.grid + ' = <b>' + fmt(s.tokens) + '</b> tokens</span>' +
        '<span class="rtd-tokbar"><i style="width:' + tokPct + '%"></i></span>' +
        '<span class="rtd-cnote">' + s.note + '</span>' +
        '<span class="rtd-chk"><input type="checkbox" data-scale="' + s.key + '"> 在此尺度跑 self-attention</span>' +
        '</label>';
    }).join('');

    var pts = DEC_AP.map(function (ap, i) { return xL(i) + ',' + apY(ap).toFixed(1); }).join(' ');
    var ticks = DEC_AP.map(function (_, i) {
      return '<text x="' + xL(i) + '" y="90" text-anchor="middle" class="rtd-svgt">' + (i + 1) + '</text>';
    }).join('');

    return '' +
      '<div class="rtd-head">🎮 互動走讀|為什麼 RT-DETR 能又準又即時</div>' +

      /* ---------- ① attention 成本計算機 ---------- */
      '<section class="rtd-sec">' +
        '<h4>① Hybrid Encoder 成本計算機:attention 該花在哪個尺度?</h4>' +
        '<p class="rtd-desc">DINO 式 encoder 把 S3+S4+S5 攤平成一長串 token 一起做 self-attention(O(N²),S3 token 最多、主導計算量,但語義最弱,CP 值極低);' +
          'RT-DETR 的 <b>AIFI</b> 只在最高層 S5 做 intra-scale attention,跨尺度融合交給純 CNN 的 <b>CCFM</b>(PAN 式 top-down + bottom-up,RepBlock)。自己勾勾看成本差多少:</p>' +
        '<div class="rtd-cards">' + cards + '</div>' +
        '<label class="rtd-concat"><input type="checkbox" class="rtd-concat-box" checked> 跨尺度 concat(攤平成同一串 token 一起做 attention:N 先相加、再平方)</label>' +
        '<div class="rtd-presets">' +
          '<button type="button" class="rtd-btn rtd-p-detr">原始 DETR 式(全勾 + concat)</button>' +
          '<button type="button" class="rtd-btn rtd-p-aifi">RT-DETR AIFI(只勾 S5)</button>' +
        '</div>' +
        '<div class="rtd-barrow"><span class="rtd-blabel">DETR 式基準</span>' +
          '<div class="rtd-track"><i class="rtd-fill rtd-fill-base rtd-nz"></i><em class="rtd-bval rtd-bval-base"></em></div></div>' +
        '<div class="rtd-barrow"><span class="rtd-blabel">目前選擇</span>' +
          '<div class="rtd-track"><i class="rtd-fill rtd-fill-cur"></i><em class="rtd-bval rtd-bval-cur"></em></div></div>' +
        '<div class="rtd-ratio"><b class="rtd-ratio-num">—</b><span class="rtd-ratio-cap"></span></div>' +
        '<div class="rtd-formula"></div>' +
        '<p class="rtd-hint"></p>' +
        '<p class="rtd-soul">💡 Hybrid Encoder 的靈魂:<b>高層語意(S5)才需要全域 attention,低層用 CNN 融合(CCFM)就夠</b> —' +
          '「貴的 attention 只花在最值得的 S5(intra-scale),便宜的 CNN 處理跨尺度融合(cross-scale)」,這個分而治之就是 encoder 提速的全部秘密。</p>' +
        '<p class="rtd-annot">※ 倍率按 token 數 N² 估算(示意),非實測 FLOPs。</p>' +
      '</section>' +

      /* ---------- ② NMS-free 延遲穩定性 ---------- */
      '<section class="rtd-sec">' +
        '<h4>② NMS-free:延遲為什麼穩定?</h4>' +
        '<p class="rtd-desc">YOLO 的「即時」有隱形稅:NMS 要先濾掉低分框、再對剩下的框兩兩比 IoU 去重 —' +
          '<b>剩餘框數越多,NMS 越慢</b>,端到端延遲隨場景浮動;RT-DETR 承襲 DETR set prediction,直接輸出固定一組 (bbox, class)。拖拖看:</p>' +
        '<div class="rtd-slider"><span>畫面中的候選框數:</span><b class="rtd-nval">100</b>' +
          '<input type="range" class="rtd-nrange" min="10" max="300" step="5" value="100" aria-label="畫面中的候選框數"></div>' +
        '<div class="rtd-barrow"><span class="rtd-blabel">YOLO + NMS</span>' +
          '<div class="rtd-track"><i class="rtd-band"></i><i class="rtd-fill rtd-fill-yolo rtd-nz"></i><em class="rtd-bval rtd-bval-yolo"></em></div></div>' +
        '<p class="rtd-cap">NMS 有 score / IoU 兩個超參數要逐資料集調;延遲隨框數增長並浮動(淺紅色帶 = 浮動範圍,示意)。</p>' +
        '<div class="rtd-barrow"><span class="rtd-blabel">RT-DETR<br>set prediction</span>' +
          '<div class="rtd-track"><i class="rtd-fill rtd-fill-rt rtd-nz"></i><em class="rtd-bval rtd-bval-rt"></em></div></div>' +
        '<p class="rtd-cap">NMS-free 端到端直接輸出:延遲確定、無超參數、對 TensorRT 等部署引擎更友好。</p>' +
        '<p class="rtd-note">平均延遲錨點由論文 T4(TensorRT FP16)實測 FPS 換算:YOLOv8-L ~71 FPS(含 NMS)≈14.1 ms、RT-DETR-R50 108 FPS ≈9.3 ms;' +
          '「延遲隨框數增長與抖動」的曲線形狀為示意,非逐框實測。</p>' +
      '</section>' +

      /* ---------- ③ decoder 層數(加碼) ---------- */
      '<section class="rtd-sec">' +
        '<h4>③ 加碼:decoder 層數 = 免重訓的速度旋鈕</h4>' +
        '<p class="rtd-desc">DETR-like 每層 decoder 都接 auxiliary head(深監督),<b>每一層都能獨立出框</b> → 推論時可直接「早退」:' +
          '例如 6 層砍到 3 層,速度↑、AP 只小降,<b>同一份權重免重訓</b>,依硬體彈性選速度檔位。</p>' +
        '<div class="rtd-slider"><span>decoder 層數:</span><b class="rtd-dval">6</b>' +
          '<input type="range" class="rtd-drange" min="1" max="6" step="1" value="6" aria-label="decoder 層數"></div>' +
        '<div class="rtd-barrow"><span class="rtd-blabel">相對延遲</span>' +
          '<div class="rtd-track"><i class="rtd-fill rtd-fill-dec rtd-nz"></i><em class="rtd-bval rtd-bval-dec"></em></div></div>' +
        '<svg class="rtd-curve" viewBox="0 0 260 96" role="img" aria-label="decoder 層數與相對 AP 的飽和曲線(示意)">' +
          '<text x="6" y="12" class="rtd-svgt">相對 AP(示意):減層 → AP 緩降、高層數段飽和</text>' +
          '<line x1="24" y1="78" x2="250" y2="78" class="rtd-axis"></line>' +
          ticks +
          '<polyline class="rtd-pl" points="' + pts + '"></polyline>' +
          '<circle class="rtd-dot" r="4.5" cx="' + xL(5) + '" cy="' + apY(DEC_AP[5]).toFixed(1) + '"></circle>' +
        '</svg>' +
        '<p class="rtd-cap rtd-dcap"></p>' +
        '<p class="rtd-note">本小節數值為定性示意:論文給的是趨勢(減層→提速、AP 緩降),未附逐層實測數字。</p>' +
      '</section>';
  }

  /* ================= 初始化 ================= */
  window.initRtdetrDemo = function (container) {
    if (typeof container === 'string') container = document.querySelector(container);
    if (!container) return null;
    injectStyle();

    // 重複初始化保護:移除舊實例(其計時器會因 isConnected 檢查自行停止)
    Array.prototype.forEach.call(container.querySelectorAll('.rtd-demo'), function (n) { n.remove(); });

    var reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

    var root = document.createElement('div');
    root.className = 'rtd-demo';
    root.innerHTML = buildHTML();
    container.appendChild(root);

    function $(sel) { return root.querySelector(sel); }

    /* ---------- ① 成本計算機 ---------- */
    var boxes = {}, cards = {};
    Array.prototype.forEach.call(root.querySelectorAll('.rtd-chk input'), function (b) { boxes[b.getAttribute('data-scale')] = b; });
    Array.prototype.forEach.call(root.querySelectorAll('.rtd-card'), function (c) { cards[c.getAttribute('data-scale')] = c; });
    var concatBox = $('.rtd-concat-box');
    var btnDetr = $('.rtd-p-detr'), btnAifi = $('.rtd-p-aifi');
    var baseFill = $('.rtd-fill-base'), baseVal = $('.rtd-bval-base');
    var curFill = $('.rtd-fill-cur'), curVal = $('.rtd-bval-cur');
    var ratioNum = $('.rtd-ratio-num'), ratioCap = $('.rtd-ratio-cap');
    var formulaEl = $('.rtd-formula'), hintEl = $('.rtd-hint');

    baseVal.textContent = fmt(BASE_COST) + '(= 8400²)';

    function updateCost() {
      var checked = SCALES.filter(function (s) { return boxes[s.key].checked; });
      var concat = concatBox.checked;
      var cost = 0, formula;

      if (!checked.length) {
        formula = '未勾選任何尺度 → 成本 = 0(純 CNN,無全域 attention)';
      } else if (concat && checked.length > 1) {
        var N = checked.reduce(function (a, s) { return a + s.tokens; }, 0);
        cost = N * N;
        formula = 'N = ' + checked.map(function (s) { return s.tokens; }).join('+') + ' = ' + fmt(N) + ' → 成本 = N² = ' + fmt(cost);
      } else if (checked.length === 1) {
        cost = checked[0].tokens * checked[0].tokens;
        formula = 'N = ' + checked[0].tokens + '(只 ' + checked[0].name + ')→ 成本 = N² = ' + fmt(cost);
      } else {
        cost = checked.reduce(function (a, s) { return a + s.tokens * s.tokens; }, 0);
        formula = '各尺度分開做:' + checked.map(function (s) { return s.tokens + '²'; }).join(' + ') + ' = ' + fmt(cost);
      }
      if (cost > 0 && cost !== BASE_COST) {
        formula += ';基準 ' + fmt(BASE_COST) + ' ÷ ' + fmt(cost) + ' ≈ ' + ratioStr(BASE_COST / cost) + '×';
      }
      formulaEl.textContent = formula;

      // 目前選擇長條(線性比例;極小值靠 min-width 保持可見)
      var frac = cost / BASE_COST;
      curFill.style.width = (frac * 100) + '%';
      curFill.classList.toggle('rtd-nz', cost > 0);
      curFill.classList.remove('rtd-c-red', 'rtd-c-orange', 'rtd-c-green');
      curFill.classList.add(frac > 0.33 ? 'rtd-c-red' : (frac > 0.02 ? 'rtd-c-orange' : 'rtd-c-green'));
      curVal.textContent = fmt(cost);

      // 倍率
      ratioNum.classList.remove('rtd-r-red', 'rtd-r-orange', 'rtd-r-green');
      if (!cost) {
        ratioNum.textContent = '—';
        ratioNum.classList.add('rtd-r-green');
        ratioCap.textContent = '無 attention 成本,但也沒有全域 context(退回純 CNN encoder)';
      } else if (cost === BASE_COST) {
        ratioNum.textContent = '1×';
        ratioNum.classList.add('rtd-r-red');
        ratioCap.textContent = '就是原始 DETR 式基準(8400² = ' + fmt(BASE_COST) + ')';
      } else {
        var r = BASE_COST / cost;
        ratioNum.textContent = '省 ≈' + ratioStr(r) + '×';
        ratioNum.classList.add(r >= 50 ? 'rtd-r-green' : (r >= 2 ? 'rtd-r-orange' : 'rtd-r-red'));
        ratioCap.textContent = '相對「原始 DETR 式(全尺度 concat)」基準';
      }
      if (!reduceMotion) {
        ratioNum.classList.remove('rtd-pulse');
        void ratioNum.offsetWidth; // 重觸發動畫
        ratioNum.classList.add('rtd-pulse');
      }

      // 動態提示(取自 rtdetr.md「Efficient Hybrid Encoder」節)
      var hasLow = checked.some(function (s) { return s.key !== 's5'; });
      var onlyS5 = checked.length === 1 && checked[0].key === 's5';
      hintEl.classList.remove('rtd-hint-good', 'rtd-hint-warn');
      if (onlyS5) {
        hintEl.classList.add('rtd-hint-good');
        hintEl.textContent = '✅ 這就是 AIFI:S5 已含完整的物件語義、token 又最少(400 vs 6400),用最小成本提煉概念之間的全域 context;跨尺度融合交給 CCFM(純 CNN)就夠。';
      } else if (hasLow) {
        hintEl.classList.add('rtd-hint-warn');
        hintEl.textContent = '⚠️ 論文明說:對低層(S3/S4)做 intra-scale attention「不僅貴,還可能有害」— 低層缺語義、特徵間關聯弱,硬做 attention 反而引入冗餘與混淆。';
      } else {
        hintEl.textContent = '目前沒有任何全域 attention,等於退回純 CNN encoder;RT-DETR 的答案:全域 attention 留一點點、放在最值得的 S5。';
      }

      // 卡片高亮 + 預設情境按鈕狀態
      SCALES.forEach(function (s) { cards[s.key].classList.toggle('rtd-on', boxes[s.key].checked); });
      btnDetr.classList.toggle('rtd-btn-on', checked.length === SCALES.length && concat);
      btnAifi.classList.toggle('rtd-btn-on', onlyS5);
    }

    function setPreset(name) {
      SCALES.forEach(function (s) { boxes[s.key].checked = (name === 'detr') ? true : (s.key === 's5'); });
      concatBox.checked = (name === 'detr');
      updateCost();
    }

    SCALES.forEach(function (s) { boxes[s.key].addEventListener('change', updateCost); });
    concatBox.addEventListener('change', updateCost);
    btnDetr.addEventListener('click', function () { setPreset('detr'); });
    btnAifi.addEventListener('click', function () { setPreset('aifi'); });

    // 初始:先呈現「原始 DETR 式」的痛,基準條進場動畫
    setPreset('detr');
    if (reduceMotion) {
      baseFill.style.width = '100%';
    } else {
      requestAnimationFrame(function () { requestAnimationFrame(function () { baseFill.style.width = '100%'; }); });
    }

    /* ---------- ② NMS-free 延遲 ---------- */
    var nRange = $('.rtd-nrange'), nVal = $('.rtd-nval');
    var yoloFill = $('.rtd-fill-yolo'), yoloVal = $('.rtd-bval-yolo'), band = $('.rtd-band');
    var rtFill = $('.rtd-fill-rt'), rtVal = $('.rtd-bval-rt');

    function latPct(ms) { return Math.max(0, Math.min(100, ms / LAT_MAX * 100)) + '%'; }
    function yoloMean(n) { return YOLO_MS + (n - 145) * 0.028; } // n=145 時 ≈ 論文平均 14.1ms;斜率為示意
    function yoloAmp(n) { return 0.4 + 0.011 * n; }              // 抖動幅度(示意):框越多越不確定

    rtFill.style.width = latPct(RT_MS);
    rtVal.textContent = '≈' + RT_MS.toFixed(1) + ' ms(固定,免 NMS)';

    function updateNms(rand) {
      var n = +nRange.value;
      nVal.textContent = n;
      var mean = yoloMean(n), amp = yoloAmp(n);
      var ms = mean + (rand || 0) * amp;
      yoloFill.style.width = latPct(ms);
      yoloVal.textContent = '≈' + ms.toFixed(1) + ' ms(±' + amp.toFixed(1) + ' 浮動)';
      band.style.left = latPct(Math.max(0, mean - amp));
      band.style.width = (2 * amp / LAT_MAX * 100) + '%';
    }

    nRange.addEventListener('input', function () { updateNms(reduceMotion ? 0 : (Math.random() * 2 - 1)); });
    updateNms(0);

    var jitterTimer = null;
    if (!reduceMotion) { // 減少動態偏好時:不跑連續抖動,浮動範圍改由靜態色帶與 ±數字表達
      jitterTimer = setInterval(function () {
        if (!root.isConnected) { clearInterval(jitterTimer); return; }
        updateNms(Math.random() * 2 - 1);
      }, 620);
    }

    /* ---------- ③ decoder 層數 ---------- */
    var dRange = $('.rtd-drange'), dVal = $('.rtd-dval'), dCap = $('.rtd-dcap');
    var decFill = $('.rtd-fill-dec'), decVal = $('.rtd-bval-dec');
    var dot = $('.rtd-dot');

    function updateDec() {
      var L = +dRange.value;
      dVal.textContent = L;
      decFill.style.width = DEC_LAT[L - 1] + '%';
      decVal.textContent = '≈' + DEC_LAT[L - 1] + '%(示意)';
      dot.setAttribute('cx', xL(L - 1));
      dot.setAttribute('cy', apY(DEC_AP[L - 1]).toFixed(1));
      dCap.textContent = L + ' 層:相對延遲 ≈' + DEC_LAT[L - 1] + '%、相對 AP ≈' + DEC_AP[L - 1].toFixed(1) +
        '(定性示意)' + (L === 6 ? ' — 基準檔位' : (L === 3 ? ' — 論文舉例:6 層砍到 3 層,速度↑、AP 只小降' : ''));
    }

    dRange.addEventListener('input', updateDec);
    updateDec();

    return root;
  };
})();
