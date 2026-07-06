/*!
 * demo-rfdetr.js — RF-DETR 互動走讀 demo(自包含 vanilla JS,無外部資源)
 * 介面:window.initRfdetrDemo(container)
 * 事實來源:papers/rfdetr.md;凡標「示意」者為定性 / 概念呈現,非論文實測數字。
 */
(function () {
  'use strict';

  var STYLE_ID = 'rfn-demo-style';

  /* ================= 論文事實(取自 rfdetr.md) ================= */
  // COCO 偵測(Table 2,T4 TensorRT10 FP16)— md 表中僅列 N / S / 2XL 三個 RF-DETR 尺寸
  var RF = [
    { key: 'N',   name: 'RF-DETR (N)',   lat: 2.5,  ap: 48.0, params: '30.5M',  gflops: '31.9',
      note: 'md:+5.3 AP vs D-FINE-N(且 AP_L 70.0)' },
    { key: 'S',   name: 'RF-DETR (S)',   lat: 3.5,  ap: 52.9, params: '32.1M',  gflops: '59.8',
      note: 'md:比 RT-DETR-R18(49.0 @ 4.4ms)又準又快' },
    { key: '2XL', name: 'RF-DETR (2XL)', lat: 17.2, ap: 60.1, params: '126.9M', gflops: '438.4',
      note: 'md:60.1 AP = 第一個 COCO 破 60 的實時偵測器' }
  ];
  // 同表有數字的 baseline(md 提到 YOLOv8/v11 等,但未給數字,故不畫)
  var BASE = [
    { name: 'RT-DETR (R18)', lat: 4.4, ap: 49.0 },
    { name: 'D-FINE (N)',    lat: 1.9, ap: 42.7 }
  ];
  // NAS 消融(Table 5):LW-DETR(M) 52.6 → +DINOv2(+2%) → +O365 預訓 → +NAS → 54.6(不增延遲)

  /* ================= 小工具 ================= */
  function f1(n) { return n.toFixed(1); }

  // 圖幾何:x=延遲 0–19ms、y=AP 40–63
  var W = 520, H = 300, PL = 44, PR = 12, PT = 18, PB = 34;
  function X(lat) { return PL + lat / 19 * (W - PL - PR); }
  function Y(ap)  { return PT + (63 - ap) / 23 * (H - PT - PB); }

  // Pareto 曲線:RF 三點間以 log(延遲) 線性內插(僅作視覺帶,曲線本身即「一次訓練導出」概念)
  function apPareto(lat) {
    if (lat <= RF[0].lat) return RF[0].ap;
    if (lat >= RF[2].lat) return RF[2].ap;
    var lo = lat < RF[1].lat ? RF[0] : RF[1];
    var hi = lat < RF[1].lat ? RF[1] : RF[2];
    var t = (Math.log(lat) - Math.log(lo.lat)) / (Math.log(hi.lat) - Math.log(lo.lat));
    return lo.ap + t * (hi.ap - lo.ap);
  }

  // 固定種子 PRNG(mulberry32)→ 配置雲每次 render 一致
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  // 「數千種配置」散點雲(示意):散在 Pareto 曲線下方
  var CLOUD = (function () {
    var r = mulberry32(20260706), pts = [], i, t, lat, drop;
    for (i = 0; i < 120; i++) {
      t = r();
      lat = Math.exp(Math.log(2.5) + t * (Math.log(17.2) - Math.log(2.5)));
      drop = Math.pow(r(), 1.6) * 4.2 + 0.15;  // 多數貼近曲線(Pareto=上緣)
      pts.push({ lat: lat, ap: apPareto(lat) - drop });
    }
    return pts;
  })();

  /* ================= 樣式(注入一次,全部 rfn- 前綴) ================= */
  var CSS = '' +
    '.rfn-demo{--rfn-accent:var(--accent,#0969da);--rfn-red:var(--red,#e93147);--rfn-green:var(--green,#08b94e);--rfn-purple:var(--purple,#7852ee);--rfn-ink:var(--ink,#1f2328);--rfn-muted:var(--muted,#57606a);--rfn-line:var(--line,#d0d7de);--rfn-bg:var(--bg,#f6f8fa);' +
      'border:1px solid var(--rfn-line);border-radius:14px;background:linear-gradient(180deg,#fbfcfd,#fff);padding:16px 16px 18px;margin:1.4em 0;color:var(--rfn-ink);' +
      'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;font-size:.92rem;line-height:1.7}' +
    '.rfn-demo,.rfn-demo *{box-sizing:border-box}' +
    '.rfn-head{display:inline-block;font-weight:700;font-size:.95rem;color:var(--rfn-accent);background:rgba(9,105,218,.09);border-radius:12px;padding:6px 14px;margin:0 0 6px}' +
    '.rfn-lead{margin:.2em 0 .4em;font-size:.86rem;color:var(--rfn-muted)}' +
    '.rfn-sec{margin-top:16px}' +
    '.rfn-sec+.rfn-sec{border-top:1px dashed var(--rfn-line);padding-top:16px;margin-top:20px}' +
    '.rfn-sec h4{margin:0 0 .4em;font-size:1.03rem;line-height:1.5}' +
    '.rfn-desc{margin:.2em 0 .8em;font-size:.86rem;color:var(--rfn-muted)}' +
    /* Pareto 圖 */
    '.rfn-chart{width:100%;height:auto;display:block;border:1px solid var(--rfn-line);border-radius:12px;background:#fff}' +
    '.rfn-chart .rfn-pt{cursor:pointer;transition:r .2s}' +
    '.rfn-chart .rfn-halo circle{transition:cx .22s,cy .22s}' +
    '.rfn-legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:8px 0 2px;font-size:.76rem;color:var(--rfn-muted)}' +
    '.rfn-legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}' +
    /* 預算滑桿 */
    '.rfn-slider{display:flex;align-items:center;gap:10px;margin:12px 0 0;flex-wrap:wrap}' +
    '.rfn-slider label{font-size:.85rem;font-weight:600;white-space:nowrap}' +
    '.rfn-slider input[type=range]{flex:1 1 140px;min-width:120px;accent-color:var(--rfn-red);height:22px;cursor:pointer;margin:0}' +
    '.rfn-slider output{font-weight:700;font-variant-numeric:tabular-nums;color:var(--rfn-red);min-width:64px;text-align:right;font-size:.9rem}' +
    /* 選中卡片 */
    '.rfn-pick{border:1.5px solid var(--rfn-purple);border-radius:12px;background:rgba(120,82,238,.05);padding:10px 12px;margin-top:10px;font-size:.86rem;transition:border-color .18s,background .18s}' +
    '.rfn-pick.rfn-none{border-color:var(--rfn-red);background:rgba(233,49,71,.05)}' +
    '.rfn-pick-name{font-weight:700;font-size:.98rem;color:var(--rfn-purple)}' +
    '.rfn-pick.rfn-none .rfn-pick-name{color:var(--rfn-red)}' +
    '.rfn-pick-tag{font-size:.78rem;color:var(--rfn-muted)}' +
    '.rfn-pick-stats{font-variant-numeric:tabular-nums;margin:2px 0}' +
    '.rfn-pick-cmp{margin:2px 0;font-weight:600;color:var(--rfn-green)}' +
    '.rfn-pick-cmp.rfn-dim{color:var(--rfn-muted);font-weight:400}' +
    '.rfn-pick-note{font-size:.78rem;color:var(--rfn-muted)}' +
    /* 傳統 vs NAS 切換 */
    '.rfn-tgl{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px}' +
    '.rfn-tgl button{flex:1 1 130px;border:1.5px solid var(--rfn-line);background:#fff;border-radius:10px;padding:7px 10px;font:inherit;font-size:.85rem;font-weight:600;color:var(--rfn-muted);cursor:pointer;transition:border-color .18s,background .18s,color .18s}' +
    '.rfn-tgl button[data-mode=trad].rfn-on{border-color:var(--rfn-red);color:var(--rfn-red);background:rgba(233,49,71,.06);box-shadow:0 2px 8px rgba(233,49,71,.12)}' +
    '.rfn-tgl button[data-mode=nas].rfn-on{border-color:var(--rfn-purple);color:var(--rfn-purple);background:rgba(120,82,238,.06);box-shadow:0 2px 8px rgba(120,82,238,.12)}' +
    '.rfn-mode-desc{margin:.2em 0 .7em;font-size:.83rem;color:var(--rfn-muted)}' +
    '.rfn-trial{display:flex;align-items:center;gap:8px;font-size:.8rem;margin:4px 0;color:var(--rfn-muted)}' +
    '.rfn-trial-name{flex:0 0 62px;font-variant-numeric:tabular-nums;white-space:nowrap}' +
    '.rfn-trial-bar{flex:1;height:8px;border-radius:999px;background:var(--rfn-bg);overflow:hidden}' +
    '.rfn-trial-bar i{display:block;height:100%;width:100%;border-radius:999px;background:var(--rfn-red);opacity:.75;animation:rfn-fill .5s ease both}' +
    '.rfn-trial-bar i.rfn-i-green{background:var(--rfn-green)}' +
    '.rfn-trial-tag{flex:0 0 auto;font-size:.72rem;white-space:nowrap}' +
    '.rfn-trial-more{font-size:.78rem;color:var(--rfn-muted);margin:6px 0 2px;line-height:1.55}' +
    '.rfn-dots{display:flex;flex-wrap:wrap;gap:3px;margin:8px 0 2px}' +
    '.rfn-dots i{width:7px;height:7px;border-radius:50%;background:var(--rfn-purple);opacity:.45;animation:rfn-pop .35s ease both}' +
    '.rfn-cost{display:flex;align-items:center;gap:4px 8px;font-size:.8rem;margin:10px 0 0;padding-top:8px;border-top:1px dashed var(--rfn-line);flex-wrap:wrap}' +
    '.rfn-cost>span:first-child{flex:0 0 auto;white-space:nowrap}' +
    '.rfn-cost b{font-variant-numeric:tabular-nums}' +
    '.rfn-cost-track{flex:1;min-width:60px;height:10px;border-radius:999px;background:var(--rfn-bg);overflow:hidden}' +
    '.rfn-cost-track i{display:block;height:100%;border-radius:999px}' +
    '.rfn-cost-anim{animation:rfn-fill 3.4s ease both}' +
    '.rfn-knobs{font-size:.78rem;color:var(--rfn-muted);margin:10px 0 0;line-height:1.6}' +
    /* NAS 消融卡片 */
    '.rfn-abl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:8px}' +
    '.rfn-abl-card{border:1.5px solid var(--rfn-line);border-radius:12px;background:#fff;padding:10px 12px}' +
    '.rfn-abl-step{font-size:.74rem;font-weight:700;color:var(--rfn-muted);letter-spacing:.02em}' +
    '.rfn-abl-val{font-size:1.25rem;font-weight:800;font-variant-numeric:tabular-nums;margin:2px 0;line-height:1.3}' +
    '.rfn-abl-val small{font-size:.72rem;font-weight:600;color:var(--rfn-muted)}' +
    '.rfn-abl-note{font-size:.75rem;color:var(--rfn-muted);line-height:1.55}' +
    '.rfn-abl-card.rfn-pp .rfn-abl-val{color:var(--rfn-purple)}' +
    '.rfn-abl-card.rfn-hl{border-color:var(--rfn-green);background:rgba(8,185,78,.05)}' +
    '.rfn-abl-card.rfn-hl .rfn-abl-val{color:var(--rfn-green)}' +
    '.rfn-foot{margin:14px 0 0;font-size:.75rem;color:var(--rfn-muted);border-top:1px dashed var(--rfn-line);padding-top:10px;line-height:1.6}' +
    '@keyframes rfn-fill{from{width:0}to{width:100%}}' +
    '@keyframes rfn-pop{from{opacity:0;transform:scale(.3)}to{opacity:.45;transform:scale(1)}}' +
    '@media (prefers-reduced-motion:reduce){.rfn-demo *,.rfn-demo *::before,.rfn-demo *::after{transition:none!important;animation:none!important}}';

  /* ================= SVG 散點圖 ================= */
  function buildSvg() {
    var s = [], i, p, x, y;
    s.push('<svg class="rfn-chart" viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" role="img" ' +
      'aria-label="RF-DETR accuracy-latency Pareto 散點圖:x 軸延遲(毫秒)、y 軸 COCO AP">');
    // y 網格 + 刻度(AP 40–60)
    for (i = 40; i <= 60; i += 5) {
      y = f1(Y(i));
      s.push('<line x1="' + PL + '" y1="' + y + '" x2="' + (W - PR) + '" y2="' + y + '" stroke="#eaeef2" stroke-width="1"/>');
      s.push('<text x="' + (PL - 6) + '" y="' + f1(Y(i) + 3.5) + '" text-anchor="end" font-size="10.5" fill="#57606a">' + i + '</text>');
    }
    // x 刻度(0–15ms)
    for (i = 0; i <= 15; i += 5) {
      x = f1(X(i));
      s.push('<line x1="' + x + '" y1="' + f1(Y(40)) + '" x2="' + x + '" y2="' + f1(Y(40) + 4) + '" stroke="#d0d7de" stroke-width="1"/>');
      s.push('<text x="' + x + '" y="' + f1(Y(40) + 15) + '" text-anchor="middle" font-size="10.5" fill="#57606a">' + i + '</text>');
    }
    s.push('<line x1="' + PL + '" y1="' + f1(Y(40)) + '" x2="' + (W - PR) + '" y2="' + f1(Y(40)) + '" stroke="#d0d7de" stroke-width="1"/>');
    s.push('<text x="' + f1((PL + W - PR) / 2) + '" y="' + (H - 4) + '" text-anchor="middle" font-size="10.5" fill="#57606a">延遲 ms(T4 TensorRT10 FP16)</text>');
    s.push('<text x="6" y="12" font-size="10.5" fill="#57606a" font-weight="700">COCO AP</text>');
    // 預算外遮罩(x > 預算 → 變灰)
    s.push('<rect class="rfn-shade" x="' + f1(X(5)) + '" y="' + PT + '" width="' + f1(W - PR - X(5)) + '" height="' + f1(Y(40) - PT) + '" fill="rgba(87,96,106,.06)"/>');
    // Pareto 曲線帶(一次訓練導出;曲線形狀為 log 內插視覺化)
    var path = '';
    for (i = 0; i <= 36; i++) {
      var lat = Math.exp(Math.log(2.5) + i / 36 * (Math.log(17.2) - Math.log(2.5)));
      path += (i ? 'L' : 'M') + f1(X(lat)) + ',' + f1(Y(apPareto(lat)));
    }
    s.push('<path d="' + path + '" fill="none" stroke="#7852ee" stroke-width="2" opacity=".3"/>');
    // 配置雲(示意)
    for (i = 0; i < CLOUD.length; i++) {
      p = CLOUD[i];
      s.push('<circle cx="' + f1(X(p.lat)) + '" cy="' + f1(Y(p.ap)) + '" r="2" fill="#7852ee" opacity=".14"/>');
    }
    // baseline(灰)
    for (i = 0; i < BASE.length; i++) {
      p = BASE[i]; x = f1(X(p.lat)); y = f1(Y(p.ap));
      s.push('<circle cx="' + x + '" cy="' + y + '" r="5.5" fill="#8b949e" stroke="#fff" stroke-width="1.5">' +
        '<title>' + p.name + ':' + f1(p.lat) + 'ms / ' + f1(p.ap) + ' AP</title></circle>');
      s.push('<text x="' + f1(X(p.lat) + 8) + '" y="' + f1(Y(p.ap) + 3.5) + '" font-size="9.5" fill="#57606a">' + p.name + '</text>');
    }
    // 光暈 + 選中環(位置由 update() 控制)
    s.push('<g class="rfn-halo"><circle class="rfn-halo-a" cx="0" cy="0" r="13" fill="rgba(120,82,238,.16)"/>' +
      '<circle class="rfn-halo-b" cx="0" cy="0" r="8.5" fill="none" stroke="#7852ee" stroke-width="2"/></g>');
    // RF-DETR(紫大點,可點擊)
    for (i = 0; i < RF.length; i++) {
      p = RF[i]; x = f1(X(p.lat)); y = f1(Y(p.ap));
      s.push('<circle class="rfn-pt" data-lat="' + p.lat + '" cx="' + x + '" cy="' + y + '" r="6.5" fill="#7852ee" stroke="#fff" stroke-width="2">' +
        '<title>' + p.name + ':' + f1(p.lat) + 'ms / ' + f1(p.ap) + ' AP(點我把預算設到這裡)</title></circle>');
      var anchorEnd = X(p.lat) > W - 60;
      s.push('<text x="' + f1(X(p.lat) + (anchorEnd ? -11 : 0)) + '" y="' + f1(Y(p.ap) - 11) +
        '" text-anchor="' + (anchorEnd ? 'end' : 'middle') + '" font-size="11" font-weight="700" fill="#7852ee">' + p.key + '</text>');
    }
    // 預算線
    s.push('<line class="rfn-bline" x1="' + f1(X(5)) + '" y1="' + PT + '" x2="' + f1(X(5)) + '" y2="' + f1(Y(40)) + '" stroke="#e93147" stroke-width="1.5" stroke-dasharray="4 3"/>');
    s.push('<text class="rfn-blabel" x="' + f1(X(5) + 5) + '" y="' + (PT + 10) + '" font-size="10.5" font-weight="700" fill="#e93147">預算 5.0ms</text>');
    s.push('</svg>');
    return s.join('');
  }

  /* ================= 傳統 / NAS 面板 ================= */
  function tradHTML() {
    var h = '<p class="rfn-mode-desc">傳統做法:想知道某一種配置(某個 patch size × 層數 × 解析度…)準不準、多快,就得<b>從頭重訓一次</b>。想掃出整條 Pareto 曲線?訓練次數 = 配置數(示意)。</p>';
    for (var i = 1; i <= 6; i++) {
      h += '<div class="rfn-trial"><span class="rfn-trial-name">配置 #' + i + '</span>' +
        '<span class="rfn-trial-bar"><i style="animation-delay:' + ((i - 1) * 0.5).toFixed(1) + 's"></i></span>' +
        '<span class="rfn-trial-tag">重訓 1 次</span></div>';
    }
    h += '<div class="rfn-trial-more">⋯ 第 7 種、第 100 種、第 1000 種配置:每種都再 +1 次完整訓練(示意)</div>' +
      '<div class="rfn-cost"><span>總訓練成本</span><span class="rfn-cost-track"><i class="rfn-cost-anim" style="width:100%;background:var(--rfn-red)"></i></span><b>≈ N×(N = 想試的配置數)</b></div>';
    return h;
  }
  function nasHTML() {
    var h = '<p class="rfn-mode-desc">weight-sharing NAS(靈感 OFA):base network <b>只訓練 1 次</b> —— 每個 iteration 隨機採一個配置做梯度更新(權重共享),等於平行訓練數千個 sub-nets(像 dropout ensemble)。<b>訓完:所有配置直接可用、不重訓</b>,整條 accuracy-latency Pareto 曲線一次到手,依硬體預算(延遲/VRAM)選一點即可。</p>' +
      '<div class="rfn-trial"><span class="rfn-trial-name">base 網路</span>' +
      '<span class="rfn-trial-bar"><i class="rfn-i-green"></i></span>' +
      '<span class="rfn-trial-tag">訓練 1 次</span></div>' +
      '<div class="rfn-dots">';
    for (var i = 0; i < 60; i++) {
      h += '<i style="animation-delay:' + (0.55 + i * 0.018).toFixed(3) + 's"></i>';
    }
    h += '</div>' +
      '<div class="rfn-trial-more">↑ 數千種配置直接評估、零重訓(此處以 60 點示意),還兼作 regularizer 提升泛化</div>' +
      '<div class="rfn-cost"><span>總訓練成本</span><span class="rfn-cost-track"><i style="width:6%;background:var(--rfn-green)"></i></span><b>= 1×</b></div>';
    return h;
  }

  /* ================= 進入點 ================= */
  window.initRfdetrDemo = function (container) {
    if (!container || !container.appendChild) return null;
    // 同容器重複 init:先清舊實例
    if (typeof container.__rfnCleanup === 'function') {
      try { container.__rfnCleanup(); } catch (e) { /* noop */ }
    }
    container.innerHTML = '';
    container.__rfnCleanup = function () { container.innerHTML = ''; };

    if (!document.getElementById(STYLE_ID)) {
      var st = document.createElement('style');
      st.id = STYLE_ID;
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    var reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

    var root = document.createElement('div');
    root.className = 'rfn-demo';
    root.innerHTML = '' +
      '<span class="rfn-head">🎮 互動走讀|weight-sharing NAS:訓練一次,數千種配置任你挑</span>' +
      '<p class="rfn-lead">RF-DETR 不是「一個模型」,而是一次訓練導出的一整條 accuracy-latency Pareto 曲線 —— 拖動下面的延遲預算,體驗「按硬體挑模型」怎麼運作。</p>' +

      '<div class="rfn-sec">' +
        '<h4>① NAS Pareto 選型器:拖「延遲預算」,自動挑最合身的 RF-DETR</h4>' +
        '<p class="rfn-desc">紫色大點 = 論文 COCO 表(Table 2)裡的 RF-DETR 三個尺寸(N / S / 2XL);灰點 = 同表 baseline;淡紫小點雲 = 一次訓練導出的數千種配置(示意)。也可以直接點紫點。</p>' +
        buildSvg() +
        '<div class="rfn-legend">' +
          '<span><i style="background:#7852ee"></i>RF-DETR(Table 2 實測)</span>' +
          '<span><i style="background:#8b949e"></i>baseline(同表)</span>' +
          '<span><i style="background:rgba(120,82,238,.3)"></i>NAS 配置雲(示意)</span>' +
        '</div>' +
        '<div class="rfn-slider">' +
          '<label for="">⏱ 延遲預算</label>' +
          '<input class="rfn-range" type="range" min="1.5" max="19" step="0.1" value="5" aria-label="延遲預算(毫秒)">' +
          '<output class="rfn-bval">5.0 ms</output>' +
        '</div>' +
        '<div class="rfn-pick" aria-live="polite"></div>' +
      '</div>' +

      '<div class="rfn-sec">' +
        '<h4>② 一次訓練 vs 每配置重訓:成本差在哪</h4>' +
        '<div class="rfn-tgl" role="group" aria-label="訓練方式切換">' +
          '<button type="button" data-mode="trad" class="rfn-on" aria-pressed="true">🏋️ 傳統做法:逐一重訓</button>' +
          '<button type="button" data-mode="nas" aria-pressed="false">✨ weight-sharing NAS</button>' +
        '</div>' +
        '<div class="rfn-mode-panel"></div>' +
        '<p class="rfn-knobs">🎛 五個可調 knobs(Fig 3):patch size(FlexiViT 插值)/ decoder 層數(可砍到 single-stage)/ query 數 / 影像解析度 / 每 block window 數 —— 推論時調這些,就是在上圖 Pareto 曲線上移動。</p>' +
      '</div>' +

      '<div class="rfn-sec">' +
        '<h4>③ NAS 消融(Table 5):每一步加多少</h4>' +
        '<div class="rfn-abl">' +
          '<div class="rfn-abl-card"><div class="rfn-abl-step">起點</div><div class="rfn-abl-val">52.6 <small>AP</small></div><div class="rfn-abl-note">LW-DETR (M) 基準(COCO)</div></div>' +
          '<div class="rfn-abl-card rfn-pp"><div class="rfn-abl-step">疊加改造</div><div class="rfn-abl-val">+2<small>% AP</small></div><div class="rfn-abl-note">換 DINOv2 backbone(取代 CAEv2)先 +2%;再加 O365 預訓與 weight-sharing NAS(md 未列逐步中間值)</div></div>' +
          '<div class="rfn-abl-card rfn-hl"><div class="rfn-abl-step">終點(全套)</div><div class="rfn-abl-val">54.6 <small>AP</small></div><div class="rfn-abl-note">md:「NAS 讓精度 +2% over LW-DETR 而不增延遲」</div></div>' +
        '</div>' +
      '</div>' +

      '<p class="rfn-foot">📎 數據取自論文 COCO 表(Table 2,T4 TensorRT10 FP16)與 NAS 消融(Table 5);淡紫散點雲、訓練成本條與動畫為概念示意,非論文實測數字。</p>';

    container.appendChild(root);

    function $(sel) { return root.querySelector(sel); }

    /* ---------- ① 延遲預算滑桿 ---------- */
    var range = $('.rfn-range'), bval = $('.rfn-bval'), pick = $('.rfn-pick');
    var bline = $('.rfn-bline'), blabel = $('.rfn-blabel'), shade = $('.rfn-shade');
    var halo = $('.rfn-halo'), haloA = $('.rfn-halo-a'), haloB = $('.rfn-halo-b');
    var ptEls = root.querySelectorAll('.rfn-pt');

    function update() {
      var b = parseFloat(range.value);
      bval.textContent = b.toFixed(1) + ' ms';

      // 預算線 + 遮罩
      var bx = X(b);
      bline.setAttribute('x1', f1(bx)); bline.setAttribute('x2', f1(bx));
      var flip = bx > W - 92;
      blabel.setAttribute('x', f1(bx + (flip ? -5 : 5)));
      blabel.setAttribute('text-anchor', flip ? 'end' : 'start');
      blabel.textContent = '預算 ' + b.toFixed(1) + 'ms';
      shade.setAttribute('x', f1(bx));
      shade.setAttribute('width', f1(Math.max(0, W - PR - bx)));

      // 預算內 AP 最高的 RF-DETR / baseline
      var rf = null, bl = null, i;
      for (i = 0; i < RF.length; i++) if (RF[i].lat <= b && (!rf || RF[i].ap > rf.ap)) rf = RF[i];
      for (i = 0; i < BASE.length; i++) if (BASE[i].lat <= b && (!bl || BASE[i].ap > bl.ap)) bl = BASE[i];

      // 點放大 + 光暈
      for (i = 0; i < ptEls.length; i++) {
        ptEls[i].setAttribute('r', rf && parseFloat(ptEls[i].getAttribute('data-lat')) === rf.lat ? 8 : 6.5);
      }
      if (rf) {
        halo.style.display = '';
        haloA.setAttribute('cx', f1(X(rf.lat))); haloA.setAttribute('cy', f1(Y(rf.ap)));
        haloB.setAttribute('cx', f1(X(rf.lat))); haloB.setAttribute('cy', f1(Y(rf.ap)));
      } else {
        halo.style.display = 'none';
      }

      // 選中卡片
      if (rf) {
        var cmp;
        if (bl) {
          cmp = '<div class="rfn-pick-cmp">vs 同預算最佳 baseline ' + bl.name + '(' + f1(bl.ap) + ' AP @ ' + f1(bl.lat) + 'ms)→ 領先 <b>+' + (rf.ap - bl.ap).toFixed(1) + ' AP</b></div>';
        } else {
          cmp = '<div class="rfn-pick-cmp rfn-dim">此預算內表中沒有 baseline 進得來(最快的 D-FINE-N 要 1.9 ms)</div>';
        }
        pick.className = 'rfn-pick';
        pick.innerHTML = '<span class="rfn-pick-name">🎯 ' + rf.name + '</span><span class="rfn-pick-tag"> — 預算內 AP 最高的配置</span>' +
          '<div class="rfn-pick-stats">延遲 <b>' + f1(rf.lat) + ' ms</b>|COCO AP <b>' + f1(rf.ap) + '</b>|參數 ' + rf.params + '|' + rf.gflops + ' GFLOPs</div>' +
          cmp + '<div class="rfn-pick-note">' + rf.note + '</div>';
      } else {
        pick.className = 'rfn-pick rfn-none';
        pick.innerHTML = '<span class="rfn-pick-name">⛔ 預算 ' + b.toFixed(1) + ' ms:RF-DETR 塞不進</span>' +
          '<div class="rfn-pick-stats">表中最快的 RF-DETR (N) 也要 2.5 ms —— 把預算往右拖一點。</div>' +
          (bl ? '<div class="rfn-pick-note">此預算內只有 baseline D-FINE (N)(1.9ms / 42.7 AP);拉到 2.5ms 後 RF-DETR (N) 就以 48.0 AP(+5.3)接手。</div>'
              : '<div class="rfn-pick-note">表中所有模型都塞不進這個預算。</div>');
      }
    }

    range.addEventListener('input', update);
    // 點紫點 = 把預算設到該配置的延遲
    for (var k = 0; k < ptEls.length; k++) {
      ptEls[k].addEventListener('click', function () {
        range.value = this.getAttribute('data-lat');
        update();
      });
    }
    update();

    /* ---------- ② 傳統 vs NAS ---------- */
    var panel = $('.rfn-mode-panel');
    var tglBtns = root.querySelectorAll('.rfn-tgl button');
    function setMode(mode) {
      for (var i = 0; i < tglBtns.length; i++) {
        var on = tglBtns[i].getAttribute('data-mode') === mode;
        tglBtns[i].className = on ? 'rfn-on' : '';
        tglBtns[i].setAttribute('aria-pressed', on ? 'true' : 'false');
      }
      // 重新注入 HTML → CSS 動畫重跑(reduced-motion 時由 media query 直接靜態化)
      panel.innerHTML = mode === 'trad' ? tradHTML() : nasHTML();
    }
    for (var m = 0; m < tglBtns.length; m++) {
      tglBtns[m].addEventListener('click', function () { setMode(this.getAttribute('data-mode')); });
    }
    setMode('trad');
    // reduceMotion 交由 CSS media query 處理;此處僅保留變數供未來 JS 動畫判斷
    void reduceMotion;

    return root;
  };
})();
