/* ==========================================================================
   demo-yoloe.js — 「YOLOE 零開銷三模式」互動 demo(自包含模組)
   用法:
     <div id="yoloe-demo"></div>
     <script src="papers/demo-yoloe.js"></script>
     <script>initYoloeDemo(document.getElementById('yoloe-demo'))</script>
   規格:
     - window.initYoloeDemo(container):在 container 內建立整個 demo
     - 樣式僅注入一次(#yde-style),class 全部 yde- 前綴,不污染全域
     - 無外部資源;支援 ~350px 窄容器(管線圖橫向可滑);prefers-reduced-motion 關閉動畫
   數據來源:papers/yoloe.md(TL;DR、§3 三模組深入、Table 1 / 5 / 6 / 7);管線圖為教學示意
   ========================================================================== */
(function () {
  'use strict';

  var STYLE_ID = 'yde-style';
  var MODES = ['text', 'visual', 'free'];

  /* ---------- 樣式(注入一次) ---------- */
  var CSS = `
.yde-root{
  --yde-ink:#1f2328;--yde-muted:#57606a;--yde-line:#d0d7de;--yde-accent:#0969da;--yde-bg:#f6f8fa;
  --yde-orange:#ec7500;--yde-yellow:#e0ac00;--yde-green:#08b94e;--yde-teal:#00bfbc;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;
  color:var(--yde-ink);background:#fff;border:1px solid var(--yde-line);border-radius:16px;
  box-shadow:0 8px 24px rgba(0,0,0,.06);padding:18px 18px 20px;line-height:1.7;
}
.yde-title{margin:0 0 2px;font-size:1.02rem;font-weight:800}
.yde-guide{margin:0;font-size:.84rem;color:var(--yde-muted)}
/* ---- 管線圖 ---- */
.yde-pipe{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:14px 0 4px}
.yde-svg{display:block;min-width:560px;width:100%;height:auto;aspect-ratio:660/250;font-family:inherit}
.yde-svg text{user-select:none;-webkit-user-select:none;pointer-events:none}
.yde-nt{font-size:13px;font-weight:700;fill:var(--yde-ink)}
.yde-ns{font-size:10.5px;fill:var(--yde-muted)}
.yde-bt{font-size:13px;font-weight:700}
.yde-bs{font-size:10px}
.yde-wgray{fill:none;stroke:#8b949e;stroke-width:2}
.yde-wire{fill:none;stroke-width:2;transition:opacity .4s ease}
.yde-wire.yde-sel{stroke-width:2.6}
.yde-wire.yde-dim{opacity:.15}
.yde-branch{cursor:pointer;transform-box:fill-box;transform-origin:center;transition:transform .55s ease,opacity .5s ease}
.yde-branch.yde-dim{opacity:.24}
.yde-branch.yde-sel .yde-brect{stroke-width:2.6;filter:drop-shadow(0 3px 5px rgba(31,35,40,.22))}
/* 推論模式:分支折疊進 head、訓練連線淡出、直通線與徽章亮起 */
.yde-root.yde-infer .yde-branch{opacity:0;pointer-events:none}
.yde-root.yde-infer .yde-branch[data-mode="text"]{transform:translate(188px,80px) scale(.04)}
.yde-root.yde-infer .yde-branch[data-mode="visual"]{transform:translate(188px,0px) scale(.04)}
.yde-root.yde-infer .yde-branch[data-mode="free"]{transform:translate(188px,-80px) scale(.04)}
.yde-root.yde-infer .yde-wire-train{opacity:0}
.yde-direct{fill:none;stroke:var(--yde-green);stroke-width:3;opacity:0;stroke-dasharray:290;stroke-dashoffset:290;transition:opacity .3s ease .25s,stroke-dashoffset .55s ease .3s}
.yde-root.yde-infer .yde-direct{opacity:1;stroke-dashoffset:0}
.yde-zbadge{opacity:0;pointer-events:none;transform:scale(.5);transform-box:fill-box;transform-origin:center;transition:opacity .3s ease .45s,transform .5s cubic-bezier(.34,1.56,.64,1) .45s}
.yde-root.yde-infer .yde-zbadge{opacity:1;transform:scale(1)}
.yde-hrect,.yde-bbrect{transition:stroke-width .3s,filter .4s}
.yde-root.yde-infer .yde-hrect,.yde-root.yde-infer .yde-bbrect{stroke-width:2.6;filter:drop-shadow(0 0 5px rgba(8,185,78,.5))}
/* ---- 控制列 ---- */
.yde-controls{display:flex;flex-direction:column;gap:10px;margin:8px 0 14px}
.yde-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.yde-ctl-lab{font-size:.78rem;font-weight:700;color:var(--yde-muted)}
.yde-mbtn{appearance:none;font-family:inherit;font-size:.88rem;border:1.5px solid var(--yde-line);background:#fff;color:var(--yde-ink);padding:7px 14px;border-radius:999px;cursor:pointer;transition:all .18s ease}
.yde-mbtn:hover{transform:translateY(-1px);border-color:var(--yde-accent)}
.yde-mbtn.yde-on{font-weight:700}
.yde-mbtn[data-mode="text"].yde-on{background:rgba(236,117,0,.14);border-color:var(--yde-orange);color:#b35a00}
.yde-mbtn[data-mode="visual"].yde-on{background:rgba(224,172,0,.16);border-color:var(--yde-yellow);color:#8a6a00}
.yde-mbtn[data-mode="free"].yde-on{background:rgba(0,191,188,.13);border-color:var(--yde-teal);color:#007a78}
.yde-seg{display:inline-flex;gap:4px;padding:4px;border:1.5px solid var(--yde-line);border-radius:999px;background:#fff;flex-wrap:wrap}
.yde-sbtn{appearance:none;border:0;background:transparent;font-family:inherit;font-size:.92rem;font-weight:600;color:var(--yde-muted);padding:8px 15px;border-radius:999px;cursor:pointer;transition:all .18s ease}
.yde-sbtn[data-phase="train"].yde-on{background:rgba(9,105,218,.12);color:var(--yde-accent);box-shadow:inset 0 0 0 1.5px var(--yde-accent)}
.yde-sbtn[data-phase="infer"].yde-on{background:rgba(8,185,78,.14);color:#067d38;box-shadow:inset 0 0 0 1.5px var(--yde-green)}
/* ---- 說明面板 ---- */
.yde-explain{border-left:4px solid var(--yde-accent);background:var(--yde-bg);border-radius:0 10px 10px 0;padding:10px 14px;font-size:.86rem;color:#39424d;min-height:84px;margin:0 0 16px;transition:border-color .3s}
.yde-explain strong{color:var(--yde-ink)}
.yde-fade{animation:ydeFade .35s ease}
@keyframes ydeFade{from{opacity:.2;transform:translateY(5px)}to{opacity:1;transform:none}}
/* ---- 數字對比卡 ---- */
.yde-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 10px}
.yde-card{border:1px solid var(--yde-line);border-radius:12px;background:linear-gradient(180deg,#fbfcfd,#fff);padding:12px 10px;text-align:center;transition:box-shadow .2s,transform .2s}
.yde-card:hover{box-shadow:0 8px 20px rgba(0,0,0,.09);transform:translateY(-2px)}
.yde-num{font-size:1.45rem;font-weight:800;line-height:1.25}
.yde-lab{font-size:.8rem;font-weight:600;margin-top:2px}
.yde-sub{font-size:.72rem;color:var(--yde-muted);margin-top:2px}
.yde-foot{margin:0;font-size:.72rem;color:var(--yde-muted)}
@media (max-width:420px){
  .yde-root{padding:14px 12px 16px}
  .yde-mbtn{font-size:.82rem;padding:6px 11px}
  .yde-sbtn{font-size:.85rem;padding:7px 11px}
}
@media (prefers-reduced-motion:reduce){
  .yde-root *,.yde-root *::before,.yde-root *::after{transition-duration:.01ms!important;transition-delay:0ms!important;animation-duration:.01ms!important;animation-delay:0ms!important}
}
`;

  /* ---------- 文案(取自 yoloe.md §3 三模組深入 / §2 折疊精神) ---------- */
  var TEXTS = {
    text: {
      name: '📝 Text prompt — RepRTA(Re-parameterizable Region-Text Alignment)',
      color: '#ec7500',
      body: 'CLIP text encoder 先把類別文字編成 embedding,而且訓練前全部 cache 好,text encoder 可整個移除、零額外訓練成本。訓練時掛一個只有單一 SwiGLU FFN block 的輕量輔助網路精煉文字 embedding、強化區域-文字對齊;訓練後 re-parameterize 折進 object embedding head 的最後一層 conv,架構與原 YOLO 完全相同。消融:+RepRTA 帶來 +2.3 AP、零推論開銷(Table 5)。'
    },
    visual: {
      name: '🖼️ Visual prompt — SAVPE(Semantic-Activated Visual Prompt Encoder)',
      color: '#e0ac00',
      body: '用 box/mask 指定「找長這樣的」:語意分支輸出與 prompt 無關的語意特徵(是什麼),激活分支把 visual prompt 形式化成 mask、與影像特徵融合出 prompt-aware 權重(在哪、多重要),兩者在低維(A 組通道)聚合成 visual prompt embedding,完全不用 transformer。消融:比單純 mask pooling +1.5 AP,A=16 最佳平衡(Table 6);對「講不出名字」的專門領域目標特別有用。'
    },
    free: {
      name: '🚫 Prompt-free — LRPC(Lazy Region-Prompt Contrast)',
      color: '#00bfbc',
      body: '把「找出所有物件並命名」重述成 retrieval 問題:先用一個專門找「所有物件」的 specialized prompt embedding 濾出有物件的 anchor,再只對這些 anchor lazy 比對內建大詞彙(tag list)取類別名,跳過大量無關 anchor、不勞駕 LLM(先前 GRiT 用 FlanT5、DINO-X 用 OPT)。消融:閾值 δ=10⁻² 時 YOLOE-v8-S 得 1.9× 加速、僅 0.2 AP drop(Table 7)。'
    },
    infer: {
      name: '🚀 推論時 — re-parameterize 折疊,零推論開銷',
      color: '#08b94e',
      body: '三個提示模組都遵循同一設計:訓練時掛輕量輔助網路強化,訓練後 re-parameterize 折疊回主幹——RepRTA 的 SwiGLU FFN 折進 object embedding head 的最後一層 1×1 conv,SAVPE 分支與 LRPC 的 specialized embedding 同樣收編。推論時就是一個乾淨 YOLO:零額外參數、零額外延遲、零部署/遷移開銷。'
    }
  };

  /* ---------- 數字對比卡(yoloe.md Table 1,v8-S vs YOLO-Worldv2-S) ---------- */
  var CARDS = [
    { num: '+3.5 AP', color: '#08b94e', lab: 'vs YOLO-Worldv2-S(LVIS)', sub: '27.9 vs 22.7 Fixed AP,zero-shot minival(Table 1)' },
    { num: '1/3', color: '#ec7500', lab: '訓練成本', sub: '12.0h vs 41.7h(v8-S,Table 1)' },
    { num: '1.4×', color: '#0969da', lab: '推論速度(T4 GPU)', sub: '305.8 vs 216.4 FPS(v8-S,Table 1)' }
  ];

  function marker(id, color) {
    return '<marker id="' + id + '" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">' +
      '<path d="M0 0L10 5L0 10z" fill="' + color + '"></path></marker>';
  }

  function buildHTML(uid) {
    var m = function (k) { return 'url(#' + uid + '-' + k + ')'; };
    return `
<div class="yde-root" aria-label="YOLOE 互動 demo">
  <p class="yde-title">🎮 互動走讀|三種 prompt、一個模型、零推論開銷</p>
  <p class="yde-guide">點「提示模式」看各分支怎麼把提示變成 prompt embedding;再切到「推論時」,看三個輔助模組如何 re-parameterize 折疊進主幹、變回乾淨 YOLO。</p>

  <div class="yde-pipe">
    <svg class="yde-svg" viewBox="0 0 660 250" width="660" height="250" role="img"
      aria-label="YOLOE 管線示意圖:輸入影像經 YOLOv8 backbone 與 neck,三個提示分支 RepRTA、SAVPE、LRPC 併入 head,輸出偵測與分割結果">
      <defs>
        ${marker(uid + '-mg', '#8b949e')}
        ${marker(uid + '-mo', '#ec7500')}
        ${marker(uid + '-my', '#e0ac00')}
        ${marker(uid + '-mt', '#00bfbc')}
        ${marker(uid + '-mn', '#08b94e')}
      </defs>

      <!-- 固定連線 -->
      <path class="yde-wgray" d="M80 125 L96 125" marker-end="${m('mg')}"></path>
      <path class="yde-wgray" d="M590 125 L596 125" marker-end="${m('mg')}"></path>

      <!-- 訓練時連線:backbone → 三分支 → head -->
      <path class="yde-wire yde-wire-train" data-mode="text" d="M214 112 C252 112 252 45 278 45" stroke="#ec7500" marker-end="${m('mo')}"></path>
      <path class="yde-wire yde-wire-train" data-mode="visual" d="M214 125 L278 125" stroke="#e0ac00" marker-end="${m('my')}"></path>
      <path class="yde-wire yde-wire-train" data-mode="free" d="M214 138 C252 138 252 205 278 205" stroke="#00bfbc" marker-end="${m('mt')}"></path>
      <path class="yde-wire yde-wire-train" data-mode="text" d="M426 45 C460 45 460 108 490 108" stroke="#ec7500" marker-end="${m('mo')}"></path>
      <path class="yde-wire yde-wire-train" data-mode="visual" d="M426 125 L490 125" stroke="#e0ac00" marker-end="${m('my')}"></path>
      <path class="yde-wire yde-wire-train" data-mode="free" d="M426 205 C460 205 460 142 490 142" stroke="#00bfbc" marker-end="${m('mt')}"></path>

      <!-- 推論時:折疊後的乾淨直通線 -->
      <path class="yde-direct" d="M214 125 L490 125" marker-end="${m('mn')}"></path>

      <!-- 輸入 -->
      <rect x="6" y="99" width="74" height="52" rx="10" fill="#f6f8fa" stroke="#d0d7de" stroke-width="1.5"></rect>
      <text class="yde-nt" x="43" y="130" text-anchor="middle">輸入影像</text>

      <!-- backbone + neck(固定,綠) -->
      <g>
        <rect class="yde-bbrect" x="102" y="95" width="112" height="60" rx="12" fill="rgba(8,185,78,.12)" stroke="#08b94e" stroke-width="1.5"></rect>
        <text class="yde-nt" x="158" y="116" text-anchor="middle">YOLOv8</text>
        <text class="yde-ns" x="158" y="132" text-anchor="middle">backbone + neck</text>
        <text class="yde-ns" x="158" y="146" text-anchor="middle">(固定)</text>
      </g>

      <!-- 三個 prompt 分支 -->
      <g class="yde-branch" data-mode="text">
        <rect class="yde-brect" x="286" y="21" width="140" height="48" rx="12" fill="rgba(236,117,0,.13)" stroke="#ec7500" stroke-width="1.5"></rect>
        <text class="yde-bt" x="356" y="41" text-anchor="middle" fill="#b35a00">📝 RepRTA</text>
        <text class="yde-bs" x="356" y="58" text-anchor="middle" fill="#b35a00">text prompt</text>
      </g>
      <g class="yde-branch" data-mode="visual">
        <rect class="yde-brect" x="286" y="101" width="140" height="48" rx="12" fill="rgba(224,172,0,.15)" stroke="#e0ac00" stroke-width="1.5"></rect>
        <text class="yde-bt" x="356" y="121" text-anchor="middle" fill="#8a6a00">🖼️ SAVPE</text>
        <text class="yde-bs" x="356" y="138" text-anchor="middle" fill="#8a6a00">visual prompt</text>
      </g>
      <g class="yde-branch" data-mode="free">
        <rect class="yde-brect" x="286" y="181" width="140" height="48" rx="12" fill="rgba(0,191,188,.12)" stroke="#00bfbc" stroke-width="1.5"></rect>
        <text class="yde-bt" x="356" y="201" text-anchor="middle" fill="#007a78">🚫 LRPC</text>
        <text class="yde-bs" x="356" y="218" text-anchor="middle" fill="#007a78">prompt-free</text>
      </g>

      <!-- head -->
      <g>
        <rect class="yde-hrect" x="498" y="95" width="92" height="60" rx="12" fill="rgba(9,105,218,.08)" stroke="#0969da" stroke-width="1.5"></rect>
        <text class="yde-nt" x="544" y="119" text-anchor="middle">Head</text>
        <text class="yde-ns" x="544" y="136" text-anchor="middle">box·mask·embed</text>
      </g>

      <!-- 輸出 -->
      <rect x="602" y="99" width="52" height="52" rx="10" fill="#f6f8fa" stroke="#d0d7de" stroke-width="1.5"></rect>
      <text class="yde-nt" x="628" y="131" text-anchor="middle">輸出</text>

      <!-- 零推論開銷徽章(推論時亮出) -->
      <g class="yde-zbadge">
        <rect x="286" y="102" width="140" height="46" rx="23" fill="#08b94e"></rect>
        <text x="356" y="131" text-anchor="middle" fill="#fff" font-size="15" font-weight="700">⚡ 零推論開銷</text>
      </g>
    </svg>
  </div>

  <div class="yde-controls">
    <div class="yde-row">
      <span class="yde-ctl-lab">提示模式</span>
      <button type="button" class="yde-mbtn" data-mode="text" aria-pressed="true">📝 Text prompt</button>
      <button type="button" class="yde-mbtn" data-mode="visual" aria-pressed="false">🖼️ Visual prompt</button>
      <button type="button" class="yde-mbtn" data-mode="free" aria-pressed="false">🚫 Prompt-free</button>
    </div>
    <div class="yde-row">
      <span class="yde-ctl-lab">階段</span>
      <div class="yde-seg" role="group" aria-label="訓練/推論切換">
        <button type="button" class="yde-sbtn" data-phase="train" aria-pressed="true">🏋️ 訓練時</button>
        <button type="button" class="yde-sbtn" data-phase="infer" aria-pressed="false">🚀 推論時(re-param 折疊)</button>
      </div>
    </div>
  </div>

  <div class="yde-explain" aria-live="polite"></div>

  <div class="yde-cards">
    ${CARDS.map(function (c) {
      return '<div class="yde-card"><div class="yde-num" style="color:' + c.color + '">' + c.num +
        '</div><div class="yde-lab">' + c.lab + '</div><div class="yde-sub">' + c.sub + '</div></div>';
    }).join('')}
  </div>

  <p class="yde-foot">所有數字皆取自論文:LVIS zero-shot(Fixed AP @ minival)、FPS 於 T4 GPU(Table 1);消融見 Table 5–7。管線圖為教學示意。</p>
</div>`;
  }

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement('style');
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  /* ---------- 入口 ---------- */
  window.initYoloeDemo = function (container) {
    if (typeof container === 'string') container = document.querySelector(container);
    if (!container) return;
    ensureStyle();

    var uid = 'yde-' + Math.random().toString(36).slice(2, 8);
    container.innerHTML = buildHTML(uid);

    var root = container.querySelector('.yde-root');
    var svg = root.querySelector('.yde-svg');
    var expl = root.querySelector('.yde-explain');
    var modeBtns = Array.prototype.slice.call(root.querySelectorAll('.yde-mbtn'));
    var segBtns = Array.prototype.slice.call(root.querySelectorAll('.yde-sbtn'));
    var state = { mode: 'text', phase: 'train' };

    function render(fade) {
      root.classList.toggle('yde-infer', state.phase === 'infer');
      modeBtns.forEach(function (b) {
        var on = b.getAttribute('data-mode') === state.mode;
        b.classList.toggle('yde-on', on);
        b.setAttribute('aria-pressed', String(on));
      });
      segBtns.forEach(function (b) {
        var on = b.getAttribute('data-phase') === state.phase;
        b.classList.toggle('yde-on', on);
        b.setAttribute('aria-pressed', String(on));
      });
      MODES.forEach(function (mo) {
        var sel = mo === state.mode;
        Array.prototype.forEach.call(svg.querySelectorAll('[data-mode="' + mo + '"]'), function (el) {
          el.classList.toggle('yde-sel', sel);
          el.classList.toggle('yde-dim', !sel);
        });
      });
      var info = state.phase === 'infer' ? TEXTS.infer : TEXTS[state.mode];
      expl.style.borderLeftColor = info.color;
      expl.innerHTML = '<strong>' + info.name + '</strong><br>' + info.body;
      if (fade) {
        expl.classList.remove('yde-fade');
        void expl.offsetWidth; /* 重觸發動畫 */
        expl.classList.add('yde-fade');
      }
    }

    modeBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        state.mode = b.getAttribute('data-mode');
        state.phase = 'train'; /* 選模式時自動回訓練視角,分支才看得到 */
        render(true);
      });
    });
    segBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        state.phase = b.getAttribute('data-phase');
        render(true);
      });
    });
    Array.prototype.forEach.call(svg.querySelectorAll('.yde-branch'), function (g) {
      g.addEventListener('click', function () {
        state.mode = g.getAttribute('data-mode');
        state.phase = 'train';
        render(true);
      });
    });

    render(false);
  };
})();
