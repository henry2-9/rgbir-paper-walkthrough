/* ==========================================================================
   demo-siglip2.js — 「SigLIP 2 = SigLIP 地基 + 統一配方」互動 demo(自包含模組)
   用法:
     <div id="demo-siglip2"></div>
     <script src="papers/demo-siglip2.js"></script>
     <script>initSiglip2Demo(document.getElementById('demo-siglip2'))</script>
   規格:
     - window.initSiglip2Demo(container):在 container 內建立整個 demo;重複 init 先清舊
     - 樣式僅注入一次(#sg2-demo-style),class 全部 sg2- 前綴,不污染全域
     - 無外部資源;支援 ~350px 窄容器;prefers-reduced-motion 直接呈現最終狀態
     - 釋放:container.__sg2Cleanup() 收掉所有 timer / listener
   數據來源:papers/siglip2.md
     - §方法核心「① Sigmoid 對比」:每對 (image, text) 獨立用 sigmoid 打分,
       不必大 batch all-gather → 對應本 demo「小 batch 友善」賣點
     - §方法核心 四支配方(① sigmoid / ② caption 解碼 / ③ self-distill+masked / ④ 多語)+ NaFlex
     - §Why 對照表:dense 局部特徵、定位 referring、多語、原生長寬比 各由哪支補強
   ⚠️ 相似度矩陣分數與能力長條皆為「概念示意」的確定性 demo 值,非真實模型輸出;
      精確 benchmark 請見原論文(arXiv:2502.14786)。
   ========================================================================== */
(function () {
  'use strict';

  var STYLE_ID = 'sg2-demo-style';

  /* ---------- 矩陣參數(確定性、可重現;不使用 Math.random) ---------- */
  var SOFTMAX_SCALE = 5;   // row-softmax 溫度(示意)
  var SIG_T = 10;          // sigmoid 溫度 t(示意)
  var SIG_B = -5;          // sigmoid 偏置 b(示意,SigLIP 的 b 通常為負,反映負樣本占多數)

  // 確定性 pseudo 相似度:對角線(正配對)高、其餘低。用整數雜湊而非亂數,確保每次 init 一致。
  function rawSim(i, j) {
    if (i === j) return 0.80 + ((i * 17) % 20) / 100;      // 0.80 ~ 0.99
    var h = ((i * 7 + j * 13) * 2654435761) >>> 0;         // 確定性整數雜湊
    return -0.05 + (h % 1000) / 1000 * 0.35;               // -0.05 ~ 0.30
  }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  // 某格在目前模式下算出的分數/機率(0~1)
  function cellScore(i, j, N, mode) {
    var s = rawSim(i, j);
    if (mode === 'softmax') {
      var denom = 0;
      for (var k = 0; k < N; k++) denom += Math.exp(SOFTMAX_SCALE * rawSim(i, k));
      return Math.exp(SOFTMAX_SCALE * s) / denom;
    }
    return sigmoid(SIG_T * s + SIG_B);
  }

  /* ---------- 統一配方:四道菜 + NaFlex(siglip2.md §方法核心) ---------- */
  var RECIPE = [
    { key: 'sig',    label: '① Sigmoid 對比',            desc: 'SigLIP 地基,每對獨立打分' },
    { key: 'cap',    label: '② Caption 解碼',            desc: 'LocCa 式,加細粒度語意/定位' },
    { key: 'ssl',    label: '③ Self-distill + Masked',   desc: 'SILC/TIPS 式,補 dense 局部特徵' },
    { key: 'ml',     label: '④ 多語資料',                desc: '多語 web + 去偏配方' },
    { key: 'naflex', label: 'NaFlex(可變解析度)',       desc: 'FlexiViT + NaViT,原生長寬比' }
  ];

  // 能力 ← 由哪些配方支撐(依 siglip2.md 敘述對應);值為概念示意上限(0~100 前的權重)
  var CAPS = [
    { key: 'global', name: '全域語意',              from: { sig: 55, cap: 15, ssl: 15 } },
    { key: 'retr',   name: '檢索',                  from: { sig: 50, ml: 25, cap: 10 } },
    { key: 'dense',  name: 'dense 局部特徵(分割/偵測)', from: { ssl: 70, cap: 12 } },
    { key: 'refer',  name: '定位 referring',        from: { cap: 65, ssl: 12 } },
    { key: 'multi',  name: '多語',                  from: { ml: 82 } },
    { key: 'aspect', name: '原生長寬比',            from: { naflex: 85 } }
  ];
  var CAP_COLORS = { global: '#0969da', retr: '#00bfbc', dense: '#08b94e', refer: '#7852ee', multi: '#ec7500', aspect: '#e93147' };

  /* ---------- 樣式(注入一次) ---------- */
  var CSS = `
.sg2-root{
  --sg2-ink:#1f2328;--sg2-muted:#57606a;--sg2-line:#d0d7de;--sg2-bg:#f6f8fa;
  --sg2-accent:#0969da;--sg2-green:#08b94e;--sg2-orange:#ec7500;--sg2-purple:#7852ee;
  --sg2-teal:#00bfbc;--sg2-red:#e93147;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;
  color:var(--sg2-ink);background:#fff;border:1px solid var(--sg2-line);border-radius:16px;
  box-shadow:0 8px 24px rgba(0,0,0,.06);padding:18px 18px 20px;line-height:1.6;
}
.sg2-title{margin:0 0 2px;font-size:1.02rem;font-weight:800}
.sg2-guide{margin:0 0 14px;font-size:.84rem;color:var(--sg2-muted)}
.sg2-sec{border-top:1px dashed var(--sg2-line);padding-top:14px;margin-top:16px}
.sg2-sec:first-of-type{border-top:none;padding-top:0;margin-top:0}
.sg2-h{margin:0 0 4px;font-size:.94rem;font-weight:800;display:flex;align-items:center;gap:7px}
.sg2-h .sg2-num{display:inline-flex;width:20px;height:20px;border-radius:6px;background:var(--sg2-accent);
  color:#fff;font-size:.74rem;align-items:center;justify-content:center;font-weight:800;flex:0 0 auto}
.sg2-sub{margin:0 0 10px;font-size:.8rem;color:var(--sg2-muted)}

/* ---- 模式切換鈕 ---- */
.sg2-modes{display:inline-flex;background:var(--sg2-bg);border:1px solid var(--sg2-line);
  border-radius:10px;padding:3px;gap:3px;margin-bottom:10px}
.sg2-mode{border:none;background:transparent;color:var(--sg2-muted);font:inherit;font-size:.8rem;
  font-weight:700;padding:6px 11px;border-radius:8px;cursor:pointer;transition:background .18s,color .18s}
.sg2-mode[aria-pressed="true"]{background:#fff;color:var(--sg2-accent);box-shadow:0 1px 3px rgba(0,0,0,.12)}
.sg2-mode--sig[aria-pressed="true"]{color:var(--sg2-green)}

/* ---- batch 滑桿 ---- */
.sg2-ctl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px;font-size:.82rem}
.sg2-ctl label{font-weight:700}
.sg2-ctl input[type=range]{flex:1;min-width:120px;accent-color:var(--sg2-accent);cursor:pointer}
.sg2-nval{font-weight:800;color:var(--sg2-accent);min-width:2.2em;text-align:right}
.sg2-batchnote{font-size:.8rem;margin:2px 0 12px;min-height:1.2em}
.sg2-batchnote.sg2-warn{color:var(--sg2-red);font-weight:700}
.sg2-batchnote.sg2-ok{color:var(--sg2-green);font-weight:700}
.sg2-batchnote.sg2-neutral{color:var(--sg2-muted)}

/* ---- 矩陣 + 側欄 ---- */
.sg2-mwrap{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.sg2-mcol{flex:1 1 240px;min-width:0}
.sg2-axis-t{font-size:.72rem;color:var(--sg2-muted);text-align:center;margin:0 0 3px;padding-left:26px}
.sg2-mrow{display:flex;align-items:stretch}
.sg2-axis-l{writing-mode:vertical-rl;transform:rotate(180deg);font-size:.72rem;color:var(--sg2-muted);
  text-align:center;flex:0 0 18px;display:flex;align-items:center;justify-content:center}
.sg2-grid{display:grid;gap:4px;flex:1 1 auto;min-width:0}
.sg2-cell{position:relative;aspect-ratio:1/1;border:1.5px solid transparent;border-radius:8px;
  background:var(--sg2-bg);cursor:pointer;font:inherit;font-size:.62rem;color:var(--sg2-muted);
  display:flex;align-items:center;justify-content:center;padding:0;overflow:hidden;
  transition:background-color .35s ease,border-color .18s,opacity .18s}
.sg2-cell.sg2-dim{opacity:.32}
.sg2-cell.sg2-hot{border-color:var(--sg2-ink)}
.sg2-cell .sg2-diagdot{position:absolute;top:3px;right:3px;width:6px;height:6px;border-radius:50%;
  background:var(--sg2-green);box-shadow:0 0 0 1.5px #fff}
.sg2-legend{font-size:.72rem;color:var(--sg2-muted);margin-top:7px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.sg2-legend .sg2-sw{display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--sg2-green);
  box-shadow:0 0 0 1.5px #fff,0 0 0 2.5px var(--sg2-line)}

/* ---- 側欄資訊 ---- */
.sg2-side{flex:1 1 210px;min-width:0;background:var(--sg2-bg);border:1px solid var(--sg2-line);
  border-radius:12px;padding:11px 12px;font-size:.8rem}
.sg2-side h4{margin:0 0 7px;font-size:.82rem}
.sg2-side .sg2-hint{color:var(--sg2-muted)}
.sg2-kv{display:flex;gap:6px;margin:5px 0;align-items:baseline}
.sg2-kv b{flex:0 0 auto;color:var(--sg2-muted);font-weight:600;font-size:.76rem}
.sg2-kv span{font-weight:700}
.sg2-tag{display:inline-block;font-size:.72rem;font-weight:800;padding:1px 7px;border-radius:999px}
.sg2-tag--pos{background:rgba(8,185,78,.14);color:#0a7a37}
.sg2-tag--neg{background:rgba(233,49,71,.14);color:#b3253a}
.sg2-side .sg2-diff{margin-top:8px;padding-top:8px;border-top:1px dashed var(--sg2-line);font-size:.76rem;color:var(--sg2-muted)}

/* ---- 配方開關 + 能力長條 ---- */
.sg2-recwrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
.sg2-switches{flex:1 1 200px;min-width:0;display:flex;flex-direction:column;gap:7px}
.sg2-sw-item{display:flex;align-items:flex-start;gap:9px;border:1px solid var(--sg2-line);border-radius:10px;
  padding:8px 10px;cursor:pointer;transition:border-color .18s,background .18s}
.sg2-sw-item[data-on="1"]{border-color:var(--sg2-accent);background:rgba(9,105,218,.05)}
.sg2-sw-item input{margin-top:2px;accent-color:var(--sg2-accent);cursor:pointer;flex:0 0 auto}
.sg2-sw-item .sg2-swtx{min-width:0}
.sg2-sw-item .sg2-swtx b{display:block;font-size:.82rem}
.sg2-sw-item .sg2-swtx small{color:var(--sg2-muted);font-size:.73rem}
.sg2-bars{flex:1 1 220px;min-width:0;display:flex;flex-direction:column;gap:8px}
.sg2-bar{font-size:.78rem}
.sg2-bar .sg2-blab{display:flex;justify-content:space-between;margin-bottom:3px}
.sg2-bar .sg2-blab b{font-weight:700}
.sg2-bar .sg2-blab span{color:var(--sg2-muted);font-variant-numeric:tabular-nums}
.sg2-track{height:9px;background:var(--sg2-bg);border-radius:999px;overflow:hidden}
.sg2-fill{height:100%;width:0;border-radius:999px;transition:width .5s cubic-bezier(.4,0,.2,1)}
.sg2-summary{margin-top:13px;font-size:.83rem;background:rgba(120,82,238,.07);border:1px solid rgba(120,82,238,.25);
  border-radius:10px;padding:9px 12px}
.sg2-summary b{color:var(--sg2-purple)}

.sg2-foot{margin-top:15px;font-size:.72rem;color:var(--sg2-muted);border-top:1px solid var(--sg2-line);padding-top:9px}

@media (prefers-reduced-motion:reduce){
  .sg2-cell,.sg2-fill,.sg2-mode,.sg2-sw-item{transition:none!important}
}
@media (max-width:420px){
  .sg2-root{padding:14px 13px 16px}
  .sg2-side{flex-basis:100%}
}
`;

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* ---------- HTML 骨架 ---------- */
  function buildHTML() {
    var switches = RECIPE.map(function (r) {
      return '<label class="sg2-sw-item" data-key="' + r.key + '" data-on="1">' +
        '<input type="checkbox" checked data-rk="' + r.key + '">' +
        '<span class="sg2-swtx"><b>' + r.label + '</b><small>' + r.desc + '</small></span></label>';
    }).join('');

    var bars = CAPS.map(function (c) {
      return '<div class="sg2-bar" data-cap="' + c.key + '">' +
        '<div class="sg2-blab"><b>' + c.name + '</b><span class="sg2-bval">0</span></div>' +
        '<div class="sg2-track"><div class="sg2-fill" style="background:' + CAP_COLORS[c.key] + '"></div></div></div>';
    }).join('');

    return '' +
    '<div class="sg2-root">' +
      '<h3 class="sg2-title">🎮 互動走讀|一格一格看懂 sigmoid vs softmax,再看統一配方各補什麼</h3>' +
      '<p class="sg2-guide">SigLIP 2 不是新 loss,而是在 SigLIP 的 sigmoid 地基上,把好幾道配方煮進同一個 encoder。</p>' +

      // ---- 第 1 節:矩陣實驗 ----
      '<div class="sg2-sec">' +
        '<h3 class="sg2-h"><span class="sg2-num">1</span>sigmoid vs softmax:對比矩陣實驗</h3>' +
        '<p class="sg2-sub">矩陣格子 = image<sub>i</sub> × text<sub>j</sub>,對角線(綠點)為正配對。切換模式看「一格分數受誰影響」。</p>' +
        '<div class="sg2-modes" role="group" aria-label="loss 模式">' +
          '<button type="button" class="sg2-mode sg2-mode--soft" data-mode="softmax" aria-pressed="false">softmax(CLIP)</button>' +
          '<button type="button" class="sg2-mode sg2-mode--sig" data-mode="sigmoid" aria-pressed="true">sigmoid(SigLIP / SigLIP 2)</button>' +
        '</div>' +
        '<div class="sg2-ctl">' +
          '<label for="sg2-batch">批次大小 N</label>' +
          '<input type="range" id="sg2-batch" min="3" max="8" step="1" value="5">' +
          '<span class="sg2-nval">5×5</span>' +
        '</div>' +
        '<div class="sg2-batchnote"></div>' +
        '<div class="sg2-mwrap">' +
          '<div class="sg2-mcol">' +
            '<p class="sg2-axis-t">text<sub>j</sub> →</p>' +
            '<div class="sg2-mrow">' +
              '<div class="sg2-axis-l">image<sub>i</sub> →</div>' +
              '<div class="sg2-grid" role="grid" aria-label="相似度矩陣"></div>' +
            '</div>' +
            '<div class="sg2-legend"><span class="sg2-sw"></span>對角線 = 正配對(label +1)&nbsp;·&nbsp;顏色越深 = 分數越高</div>' +
          '</div>' +
          '<div class="sg2-side"><h4>點任一格看差異</h4><p class="sg2-hint">點矩陣裡的任一格,這裡會顯示它在兩種 loss 下如何被處理。</p></div>' +
        '</div>' +
      '</div>' +

      // ---- 第 2 節:統一配方 ----
      '<div class="sg2-sec">' +
        '<h3 class="sg2-h"><span class="sg2-num">2</span>統一配方:四道菜各解鎖什麼能力</h3>' +
        '<p class="sg2-sub">勾選 / 取消每道配方,右側能力長條即時增減(純示意)。</p>' +
        '<div class="sg2-recwrap">' +
          '<div class="sg2-switches">' + switches + '</div>' +
          '<div class="sg2-bars">' + bars + '</div>' +
        '</div>' +
        '<div class="sg2-summary">一句話:SigLIP 2 <b>不是新 loss</b>,是把這些一起「煮進同一個 encoder」。</div>' +
      '</div>' +

      '<p class="sg2-foot">分數與能力長條為概念示意(確定性 demo 值,非真實模型輸出);數值/機制出處為 siglip2.md,精確 benchmark 請見原論文(arXiv:2502.14786)。</p>' +
    '</div>';
  }

  /* ---------- 入口 ---------- */
  window.initSiglip2Demo = function (container) {
    if (typeof container === 'string') container = document.querySelector(container);
    if (!container) return;
    if (typeof container.__sg2Cleanup === 'function') container.__sg2Cleanup(); // 重複 init 先清舊
    ensureStyle();
    container.innerHTML = buildHTML();

    var REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    var root = container.querySelector('.sg2-root');

    // 狀態
    var mode = 'sigmoid';
    var N = 5;
    var sel = null;               // {i,j} 或 null
    var recipe = { sig: 1, cap: 1, ssl: 1, ml: 1, naflex: 1 };

    // 事件登記(供 cleanup)
    var listeners = [];
    function on(el, ev, fn) { el.addEventListener(ev, fn); listeners.push([el, ev, fn]); }
    function cleanup() {
      listeners.forEach(function (l) { l[0].removeEventListener(l[1], l[2]); });
      listeners.length = 0;
    }
    container.__sg2Cleanup = cleanup;

    /* ===== 第 1 節 元素 ===== */
    var modeBtns = root.querySelectorAll('.sg2-mode');
    var slider = root.querySelector('#sg2-batch');
    var nval = root.querySelector('.sg2-nval');
    var batchNote = root.querySelector('.sg2-batchnote');
    var grid = root.querySelector('.sg2-grid');
    var side = root.querySelector('.sg2-side');

    // 值 → 背景色(白 → 對應主色的線性插值)
    function scoreBg(v, hex) {
      var r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
      var t = Math.max(0, Math.min(1, v));
      var mix = function (c) { return Math.round(255 + (c - 255) * t); };
      return 'rgb(' + mix(r) + ',' + mix(g) + ',' + mix(b) + ')';
    }

    function renderMatrix() {
      grid.style.gridTemplateColumns = 'repeat(' + N + ',minmax(0,1fr))';
      grid.innerHTML = '';
      for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
          var v = cellScore(i, j, N, mode);
          var isDiag = (i === j);
          var hex = isDiag ? '#08b94e' : '#0969da';
          var cell = document.createElement('button');
          cell.type = 'button';
          cell.className = 'sg2-cell';
          cell.setAttribute('data-i', i);
          cell.setAttribute('data-j', j);
          cell.setAttribute('role', 'gridcell');
          cell.setAttribute('aria-label', 'image' + (i + 1) + ' × text' + (j + 1) + ' 分數 ' + v.toFixed(2));
          cell.style.backgroundColor = scoreBg(v, hex);
          cell.style.color = (v > 0.55) ? '#fff' : 'var(--sg2-muted)';
          cell.textContent = v.toFixed(2);
          if (isDiag) { var d = document.createElement('span'); d.className = 'sg2-diagdot'; cell.appendChild(d); }
          grid.appendChild(cell);
        }
      }
      applyHighlight();
    }

    // 依 mode 與 sel 套用 highlight:softmax→整列;sigmoid→只該格
    function applyHighlight() {
      var cells = grid.querySelectorAll('.sg2-cell');
      cells.forEach(function (c) { c.classList.remove('sg2-dim', 'sg2-hot'); });
      if (!sel) return;
      cells.forEach(function (c) {
        var ci = +c.getAttribute('data-i'), cj = +c.getAttribute('data-j');
        var inSet = (mode === 'softmax') ? (ci === sel.i) : (ci === sel.i && cj === sel.j);
        var isSel = (ci === sel.i && cj === sel.j);
        if (!inSet) c.classList.add('sg2-dim');
        if (isSel) c.classList.add('sg2-hot');
      });
    }

    function renderSide() {
      if (!sel) {
        side.innerHTML = '<h4>點任一格看差異</h4><p class="sg2-hint">點矩陣裡的任一格,這裡會顯示它在兩種 loss 下如何被處理。</p>';
        return;
      }
      var i = sel.i, j = sel.j, pos = (i === j);
      var v = cellScore(i, j, N, mode);
      var s = rawSim(i, j);
      var tag = pos
        ? '<span class="sg2-tag sg2-tag--pos">label +1(正配對)</span>'
        : '<span class="sg2-tag sg2-tag--neg">label −1(負配對)</span>';

      var denomLine, diffLine;
      if (mode === 'softmax') {
        denomLine = '第 ' + (i + 1) + ' 列全部 ' + N + ' 格(row-softmax,同列互相搶機率)';
        diffLine = 'softmax 下這格的分數<b>被同列其他 ' + (N - 1) + ' 格牽動</b>——列內是零和,負樣本越多對比越強。';
      } else {
        denomLine = '只有自己(sigmoid 獨立打分,不看其他格)';
        diffLine = 'sigmoid 下這格<b>只跟自己比</b>:過 σ(t·s+b) 判「配 / 不配」,與同列其他格無關 → 小 batch 也 OK。';
      }

      side.innerHTML =
        '<h4>image<sub>' + (i + 1) + '</sub> × text<sub>' + (j + 1) + '</sub> ' + tag + '</h4>' +
        '<div class="sg2-kv"><b>原始相似度 s</b><span>' + s.toFixed(2) + '</span></div>' +
        '<div class="sg2-kv"><b>目前分數</b><span>' + v.toFixed(3) + '</span></div>' +
        '<div class="sg2-kv"><b>分母 / 參照</b><span style="font-weight:700">' + denomLine + '</span></div>' +
        '<div class="sg2-diff">' + diffLine + '</div>';
    }

    function renderBatchNote() {
      nval.textContent = N + '×' + N;
      var cls, txt;
      if (mode === 'softmax') {
        if (N <= 4) {
          cls = 'sg2-warn';
          txt = '⚠ softmax:每列只有 ' + (N - 1) + ' 個負樣本 → negatives 太少、對比訊號弱(需要大 batch)。';
        } else {
          cls = 'sg2-neutral';
          txt = 'softmax:每列有 ' + (N - 1) + ' 個負樣本,batch 越大 negatives 越足、對比越穩。';
        }
      } else {
        cls = 'sg2-ok';
        txt = '✓ sigmoid:每對獨立打分,小 batch 也 OK(承襲 SigLIP,不需大 batch all-gather)。';
      }
      batchNote.className = 'sg2-batchnote ' + cls;
      batchNote.textContent = txt;
    }

    function setMode(m) {
      mode = m;
      modeBtns.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === m)); });
      renderMatrix();
      renderSide();
      renderBatchNote();
    }

    // 綁定第 1 節事件
    modeBtns.forEach(function (b) { on(b, 'click', function () { setMode(b.getAttribute('data-mode')); }); });
    on(slider, 'input', function () {
      N = +slider.value;
      if (sel && (sel.i >= N || sel.j >= N)) sel = null; // 縮小後選取失效則清除
      renderMatrix();
      renderSide();
      renderBatchNote();
    });
    on(grid, 'click', function (e) {
      var cell = e.target.closest ? e.target.closest('.sg2-cell') : null;
      if (!cell || !grid.contains(cell)) return;
      var i = +cell.getAttribute('data-i'), j = +cell.getAttribute('data-j');
      sel = (sel && sel.i === i && sel.j === j) ? null : { i: i, j: j }; // 再點同格取消
      applyHighlight();
      renderSide();
    });

    /* ===== 第 2 節:配方 → 能力 ===== */
    var swItems = root.querySelectorAll('.sg2-sw-item');
    var bars = {};
    CAPS.forEach(function (c) { bars[c.key] = root.querySelector('.sg2-bar[data-cap="' + c.key + '"]'); });

    function renderCaps(animate) {
      CAPS.forEach(function (c) {
        var total = 0;
        for (var k in c.from) { if (recipe[k]) total += c.from[k]; }
        if (total > 100) total = 100;
        var bar = bars[c.key];
        var fill = bar.querySelector('.sg2-fill');
        var valEl = bar.querySelector('.sg2-bval');
        if (animate && !REDUCED) {
          // 讓 CSS transition 生效
          fill.style.width = total + '%';
        } else {
          fill.style.transition = 'none';
          fill.style.width = total + '%';
          // 還原 transition
          fill.offsetWidth; // reflow
          fill.style.transition = '';
        }
        valEl.textContent = total + ' 示意';
      });
    }

    swItems.forEach(function (item) {
      var cb = item.querySelector('input');
      on(cb, 'change', function () {
        recipe[cb.getAttribute('data-rk')] = cb.checked ? 1 : 0;
        item.setAttribute('data-on', cb.checked ? '1' : '0');
        renderCaps(true);
      });
    });

    /* ===== 初次渲染 ===== */
    setMode('sigmoid');   // 預設 sigmoid,順帶 renderMatrix / renderSide / renderBatchNote
    renderCaps(false);    // 初次瞬間到位(避免載入時長條抽動)
    // 讓後續切換有動畫:下一幀補一次帶動畫的長條
    if (!REDUCED && window.requestAnimationFrame) {
      var rid = requestAnimationFrame(function () { renderCaps(true); });
      listeners.push([{ removeEventListener: function () { if (window.cancelAnimationFrame) cancelAnimationFrame(rid); } }, '', null]);
    }
  };
})();
