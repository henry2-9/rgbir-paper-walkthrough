/*
 * demo-vlacfdet.js — VL-ACFDet 走讀互動 demo
 * 「RGB-IR 對齊實驗室 + 訓練期蒸餾」
 *
 * 介面:window.initVlacfdetDemo(containerElement)
 *  - 樣式只注入一次(<style id="vla-demo-style">),class 一律 vla- 前綴
 *  - 無外部資源(純 canvas + inline SVG + CSS + emoji);容器最窄 ~350px 可用
 *  - prefers-reduced-motion:SFS 對齊動畫與淡入淡出改為瞬間到位
 *  - 重複對同一容器呼叫會先清掉舊實例(rAF / listener 皆會釋放)
 *
 * 論文事實來源:papers/vlacfdet.md(機制描述與所有數字皆取自該檔)
 * 街景、融合品質條、模態可用度條皆為「概念示意」,非真實影像處理或量測。
 */
(function () {
  'use strict';

  var STYLE_ID = 'vla-demo-style';

  // ---- 常數 ----
  var CW = 480, CH = 270;            // canvas 內部解析度(CSS 縮放至容器寬)
  var HZ = 152;                      // 地平線 y
  var MAX_OFF = 20;                  // IR 偏移滑桿範圍 ±20px
  var MAX_DIST = Math.hypot(MAX_OFF, MAX_OFF); // 品質線性映射用的最大偏移距離
  var C_BLUE = '#0969da', C_GREEN = '#08b94e', C_ORANGE = '#ec7500', C_RED = '#e93147';

  // 三種情境(§2)。可用度百分比為「示意值」,非論文量測;IR 恆定呼應「熱影像不受光照影響」。
  var WEATHERS = {
    day: {
      label: '☀️ 白天', rgb: 95, ir: 88,
      cap: '☀️ <b>白天</b>:RGB 紋理、色彩齊全 —— 但好天氣只是特例,資料集若只有白天就學不到惡劣情境。',
      aria: '白天街景示意:RGB 層清晰,行人與車輛可見,IR 熱斑穩定疊加'
    },
    night: {
      label: '🌙 夜晚', rgb: 35, ir: 88,
      cap: '🌙 <b>夜晚</b>:RGB 弱光退化、行人輪廓幾乎淹沒在暗處;IR 熱斑不受光照影響,依然清楚。',
      aria: '夜晚街景示意:RGB 層昏暗,行人輪廓難辨,IR 熱斑依然清楚'
    },
    rainfog: {
      label: '🌧 雨霧', rgb: 20, ir: 88,
      cap: '🌧 <b>雨霧</b>:RGB 疊上雨紋、雜訊與霧化最嚴重;IR 依然穩定 —— 這正是自建資料集刻意涵蓋的難場景。',
      aria: '雨霧街景示意:RGB 層被雨紋雜訊與霧遮蔽,IR 熱斑依然清楚'
    }
  };

  // 消融數字卡(vlacfdet.md 消融 Table 3,自建資料集;mAP50)
  var ABL = [
    { name: '只用可見光', map: '73.4', sub: '7.0M・123.5 FPS', tag: '' },
    { name: '只用熱影像', map: '65.6', sub: '7.0M・123.5 FPS', tag: '最差→要融合' },
    { name: 'Baseline(CFT)', map: '75.91', sub: '44.5M・47.1 FPS', tag: '' },
    { name: '+ AC-CA', map: '77.89', sub: '34.7M・52 FPS', tag: '參數↓ 速度↑' },
    { name: '+ VL-CAT', map: '78.39', sub: '46.1M・46.5 FPS', tag: 'mAP75 +5.37' },
    { name: '完整版', map: '79.42', sub: '36.3M・47.6 FPS', tag: '★ 最佳' }
  ];

  // 4 個模態品質 prompt(vlacfdet.md 逐字)
  var PROMPTS = [
    'a visible image in clear weather conditions',
    'a visible image in adverse weather conditions',
    'a thermal image with clear objects and visible details',
    'a thermal image without clear objects'
  ];

  var TRAIN_TXT = '🏋️ <b>訓練期</b>:凍結的 CLIP(影像+文字編碼器)以 4 個模態品質 prompt 產生語意訊號,' +
    '經 <b>L2 蒸餾</b>(L<sub>T</sub>、L<sub>I</sub>,權重 λ 各 0.1)把「此刻哪個模態可信」' +
    '注入 backbone <b>最後三層的 channel attention</b> f<sub>c</sub>=σ(MLP(MaxPool(f)))。';
  var INFER_TXT = '🚀 <b>推論期</b>:CLIP 整塊拿掉、也不需額外資料 —— <b>無 CLIP 級開銷</b>;' +
    '留下的只有學好的輕量通道分支(參數 +3.6%、FPS 47.1→46.5,消融 Table 3)。';

  // ---- SVG 架構小圖(§3)----
  function fmt1(n) { return Math.round(n * 10) / 10; }
  function arr(x1, y1, x2, y2, dashed) {
    var dx = x2 - x1, dy = y2 - y1, len = Math.sqrt(dx * dx + dy * dy) || 1;
    var ux = dx / len, uy = dy / len, hl = 7, hw = 4.6;
    var bx = x2 - ux * hl, by = y2 - uy * hl;
    var lc = dashed ? 'vla-ard' : 'vla-ar', pc = dashed ? 'vla-ardh' : 'vla-arh';
    return '<line class="' + lc + '" x1="' + fmt1(x1) + '" y1="' + fmt1(y1) +
      '" x2="' + fmt1(bx) + '" y2="' + fmt1(by) + '"/>' +
      '<polygon class="' + pc + '" points="' + fmt1(x2) + ',' + fmt1(y2) + ' ' +
      fmt1(bx - uy * hw) + ',' + fmt1(by + ux * hw) + ' ' +
      fmt1(bx + uy * hw) + ',' + fmt1(by - ux * hw) + '"/>';
  }
  function node(x, y, w, h, cls, title, sub) {
    var cx = x + w / 2;
    var s = '<rect class="vla-nd ' + (cls || '') + '" x="' + x + '" y="' + y +
      '" width="' + w + '" height="' + h + '" rx="9"/>';
    if (sub) {
      s += '<text class="vla-ndt" x="' + cx + '" y="' + (y + h / 2 - 1) + '">' + title + '</text>' +
        '<text class="vla-nds" x="' + cx + '" y="' + (y + h / 2 + 12) + '">' + sub + '</text>';
    } else {
      s += '<text class="vla-ndt" x="' + cx + '" y="' + (y + h / 2 + 4) + '">' + title + '</text>';
    }
    return s;
  }
  var SVG =
    '<svg class="vla-svg" viewBox="0 0 460 344" xmlns="http://www.w3.org/2000/svg" role="img" ' +
      'aria-label="VL-ACFDet 架構示意:雙流 YOLOv5s 骨幹的最後三層特徵,分別經 AC-CA 對齊融合與 VL-CAT 通道注意力後進 FPN 與偵測頭;訓練期 CLIP 老師以 L2 蒸餾注入語意,推論期整塊移除">' +
      node(30, 10, 110, 30, 'vla-nd-b', 'RGB 輸入') +
      node(170, 10, 110, 30, 'vla-nd-r', 'IR 輸入') +
      node(30, 58, 110, 34, '', 'YOLOv5s 骨幹', 'RGB 分支') +
      node(170, 58, 110, 34, '', 'YOLOv5s 骨幹', 'IR 分支') +
      node(70, 116, 170, 30, '', '最後三層特徵') +
      node(30, 174, 125, 48, 'vla-nd-g', 'AC-CA', '對齊+注意力融合') +
      node(165, 174, 125, 48, 'vla-nd-o', 'VL-CAT 通道注意力', 'σ(MLP(MaxPool(f)))') +
      node(70, 248, 170, 30, '', 'FPN(P3/P4/P5)') +
      node(70, 300, 170, 32, '', '偵測頭(框+類別)') +
      arr(85, 40, 85, 58) + arr(225, 40, 225, 58) +
      arr(85, 92, 85, 116) + arr(225, 92, 225, 116) +
      arr(115, 146, 93, 174) + arr(195, 146, 220, 174) +
      arr(93, 222, 125, 248) + arr(227, 222, 185, 248) +
      arr(155, 278, 155, 300) +
      // CLIP 老師(訓練期限定,推論時整塊淡出)
      '<g class="vla-teacher">' +
        '<rect class="vla-tbox" x="308" y="70" width="142" height="164" rx="10"/>' +
        '<text class="vla-tt" x="379" y="92">❄ 凍結 CLIP 老師</text>' +
        '<rect class="vla-nd vla-nd-o" x="320" y="102" width="118" height="26" rx="8"/>' +
        '<text class="vla-ndt2" x="379" y="119">影像編碼器</text>' +
        '<rect class="vla-nd vla-nd-o" x="320" y="134" width="118" height="26" rx="8"/>' +
        '<text class="vla-ndt2" x="379" y="151">文字編碼器</text>' +
        '<text class="vla-ts" x="379" y="180">4 個模態品質 prompt</text>' +
        '<text class="vla-ts" x="379" y="194">(清晰/惡劣 × RGB/IR)</text>' +
        '<rect class="vla-tbadge" x="330" y="206" width="98" height="18" rx="9"/>' +
        '<text class="vla-tbt" x="379" y="219">只在訓練期</text>' +
        arr(308, 166, 292, 190, true) +
      '</g>' +
      // 推論模式徽章(與老師同位置,切推論時淡入)
      '<g class="vla-infbadge" aria-hidden="true">' +
        '<rect class="vla-ibox" x="308" y="108" width="142" height="104" rx="10"/>' +
        '<text class="vla-ibt" x="379" y="136">🚀 CLIP 已整塊退場</text>' +
        '<text class="vla-ibt" x="379" y="158">無 CLIP 級開銷</text>' +
        '<text class="vla-ibs" x="379" y="182">留下的通道分支照跑:</text>' +
        '<text class="vla-ibs" x="379" y="197">+3.6% 參數・FPS 47.1→46.5</text>' +
      '</g>' +
    '</svg>';

  // ---- CSS ----
  var CSS = [
    '.vla-wrap{border:1px solid #d0d7de;border-radius:14px;background:linear-gradient(180deg,#fbfcfd,#fff);overflow:hidden;margin:1.4em 0;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC","PingFang TC",sans-serif;line-height:1.6}',
    '.vla-head{padding:9px 14px;font-size:.9rem;font-weight:700;color:var(--accent,#0969da);background:rgba(9,105,218,.06);border-bottom:1px solid #d0d7de}',
    '.vla-body{padding:12px}',
    '.vla-sec{margin-top:18px;border-top:1px dashed #d0d7de;padding-top:13px}',
    '.vla-sec:first-child{margin-top:0;border-top:none;padding-top:0}',
    '.vla-sectitle{font-size:.88rem;font-weight:700;margin:0 0 8px}',
    '.vla-cv{display:block;width:100%;height:auto;background:#000;border-radius:10px}',
    '.vla-legend{font-size:.72rem;color:#57606a;margin-top:5px}',
    '.vla-mk-rgb{color:var(--accent,#0969da);font-weight:700}',
    '.vla-mk-ir{color:var(--orange,#ec7500);font-weight:700}',
    '.vla-qrow{display:flex;align-items:center;gap:7px;margin:0 0 7px}',
    '.vla-qlabel{font-size:.74rem;color:#57606a;white-space:nowrap}',
    '.vla-track{flex:1 1 auto;min-width:36px;height:10px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:999px;overflow:hidden}',
    '.vla-fill{height:100%;width:0;border-radius:999px;transition:width .18s linear,background .18s linear;min-width:2px}',
    '.vla-qval{font-size:.74rem;font-weight:700;min-width:3em;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}',
    '.vla-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin-top:10px;padding:9px 12px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:10px}',
    '.vla-ctl{display:flex;align-items:center;gap:7px;font-size:.78rem;color:#1f2328;flex:1 1 150px;min-width:145px}',
    '.vla-ctl input[type=range]{flex:1 1 60px;min-width:56px;accent-color:var(--accent,#0969da);margin:0}',
    '.vla-val{color:var(--accent,#0969da);font-weight:700;font-size:.76rem;min-width:3.4em;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}',
    '.vla-btn{appearance:none;font-family:inherit;font-size:.84rem;font-weight:600;color:#1f2328;background:#fff;border:1.5px solid #d0d7de;border-radius:999px;padding:5px 14px;cursor:pointer;transition:border-color .15s,color .15s,background .15s;flex:none}',
    '.vla-btn:hover{border-color:var(--accent,#0969da);color:var(--accent,#0969da)}',
    '.vla-btn-acc{background:var(--accent,#0969da);border-color:var(--accent,#0969da);color:#fff}',
    '.vla-btn-acc:hover{background:#0757a8;border-color:#0757a8;color:#fff}',
    '.vla-blocks{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:10px}',
    '.vla-block{border:1px solid #d0d7de;border-radius:10px;background:#fff;padding:8px 10px;font-size:.74rem;color:#39424d;min-width:0}',
    '.vla-block>b:first-child{display:block;font-size:.76rem;color:var(--accent,#0969da);margin-bottom:2px}',
    '.vla-fact{margin:10px 0 0;font-size:.8rem;color:#39424d;background:#f6f8fa;border-left:4px solid var(--green,#08b94e);border-radius:0 8px 8px 0;padding:7px 11px}',
    '.vla-scen{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 10px}',
    '.vla-scbtn{appearance:none;font-family:inherit;font-size:.82rem;font-weight:600;color:#1f2328;background:#fff;border:1.5px solid #d0d7de;border-radius:999px;padding:5px 14px;cursor:pointer;transition:border-color .15s,color .15s,background .15s}',
    '.vla-scbtn:hover{border-color:var(--accent,#0969da);color:var(--accent,#0969da)}',
    '.vla-scbtn.vla-on{background:var(--accent,#0969da);border-color:var(--accent,#0969da);color:#fff}',
    '.vla-mbars{margin-top:8px;display:grid;gap:5px}',
    '.vla-caption{margin:8px 0 0;font-size:.78rem;color:#39424d}',
    '.vla-cards{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}',
    '.vla-chip{font-size:.72rem;font-weight:600;border-radius:999px;padding:3px 10px;background:rgba(9,105,218,.07);border:1px solid rgba(9,105,218,.25);color:#0757a8}',
    '.vla-chip-g{background:rgba(8,185,78,.08);border-color:rgba(8,185,78,.3);color:#0a7a38}',
    '.vla-note{margin-top:8px;font-size:.72rem;color:#6b5218;background:#fff8ec;border:1px solid #f0d9ac;border-radius:8px;padding:6px 10px}',
    '.vla-modebar{margin-top:0}',
    '.vla-modelabel{font-size:.78rem;color:#39424d;font-weight:600}',
    '.vla-svgwrap{margin-top:10px;border:1px solid #d0d7de;border-radius:10px;background:#fff;padding:8px}',
    '.vla-svg{width:100%;max-width:520px;height:auto;display:block;margin:0 auto}',
    '.vla-nd{fill:#fff;stroke:#d0d7de;stroke-width:1.4}',
    '.vla-nd-b{stroke:#0969da;fill:rgba(9,105,218,.06)}',
    '.vla-nd-r{stroke:#e93147;fill:rgba(233,49,71,.06)}',
    '.vla-nd-g{stroke:#08b94e;fill:rgba(8,185,78,.07)}',
    '.vla-nd-o{stroke:#ec7500;fill:rgba(236,117,0,.07)}',
    '.vla-ndt{font-size:12px;font-weight:700;fill:#1f2328;text-anchor:middle}',
    '.vla-ndt2{font-size:10.5px;font-weight:600;fill:#1f2328;text-anchor:middle}',
    '.vla-nds{font-size:9.5px;fill:#57606a;text-anchor:middle}',
    '.vla-ar{stroke:#8c959f;stroke-width:1.6}',
    '.vla-arh{fill:#8c959f}',
    '.vla-ard{stroke:#ec7500;stroke-width:1.6;stroke-dasharray:5 4}',
    '.vla-ardh{fill:#ec7500}',
    '.vla-tbox{fill:rgba(236,117,0,.05);stroke:#ec7500;stroke-width:1.4;stroke-dasharray:6 4}',
    '.vla-tt{font-size:12px;font-weight:700;fill:#b45a00;text-anchor:middle}',
    '.vla-ts{font-size:9.5px;fill:#8a5a20;text-anchor:middle}',
    '.vla-tbadge{fill:rgba(236,117,0,.14)}',
    '.vla-tbt{font-size:9.5px;font-weight:700;fill:#b45a00;text-anchor:middle}',
    '.vla-ibox{fill:rgba(8,185,78,.07);stroke:#08b94e;stroke-width:1.4}',
    '.vla-ibt{font-size:12px;font-weight:700;fill:#0a7a38;text-anchor:middle}',
    '.vla-ibs{font-size:9.5px;fill:#3e6b4f;text-anchor:middle}',
    '.vla-teacher{opacity:1;transition:opacity .45s ease}',
    '.vla-infbadge{opacity:0;transition:opacity .45s ease;pointer-events:none}',
    '.vla-inf .vla-teacher{opacity:0}',
    '.vla-inf .vla-infbadge{opacity:1}',
    '.vla-modetext{margin:8px 0 0;font-size:.8rem;color:#39424d;background:#f6f8fa;border-left:4px solid var(--orange,#ec7500);border-radius:0 8px 8px 0;padding:7px 11px}',
    '.vla-prompts{margin-top:8px;font-size:.72rem;color:#57606a}',
    '.vla-prompts b{color:#1f2328}',
    '.vla-pchip{display:inline-block;margin:3px 4px 0 0;padding:2px 8px;border-radius:8px;background:#f6f8fa;border:1px solid #d0d7de;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;color:#39424d;word-break:break-word}',
    '.vla-abl{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-top:10px}',
    '.vla-acard{border:1px solid #d0d7de;border-radius:10px;background:#fff;padding:7px 10px;min-width:0}',
    '.vla-acard-hi{border-color:var(--green,#08b94e);box-shadow:0 0 0 1px var(--green,#08b94e) inset}',
    '.vla-aname{font-size:.72rem;color:#57606a}',
    '.vla-amap{font-size:1.06rem;font-weight:800;color:var(--accent,#0969da);font-variant-numeric:tabular-nums;line-height:1.25}',
    '.vla-amap small{font-size:.62rem;font-weight:600;color:#8c959f}',
    '.vla-asub{font-size:.68rem;color:#57606a;font-variant-numeric:tabular-nums}',
    '.vla-atag{display:inline-block;margin-top:3px;font-size:.64rem;font-weight:700;color:#0a7a38;background:rgba(8,185,78,.08);border-radius:999px;padding:1px 7px}',
    '.vla-src{margin-top:14px;font-size:.7rem;color:#6b5218;background:#fff8ec;border:1px solid #f0d9ac;border-radius:8px;padding:6px 10px}',
    '.vla-btn:focus-visible,.vla-scbtn:focus-visible,.vla-ctl input:focus-visible{outline:2px solid var(--accent,#0969da);outline-offset:2px}',
    '@media (prefers-reduced-motion: reduce){.vla-wrap .vla-fill,.vla-wrap .vla-btn,.vla-wrap .vla-scbtn,.vla-wrap .vla-teacher,.vla-wrap .vla-infbadge{transition:none!important}}'
  ].join('\n');

  // ---- HTML ----
  function ablCards() {
    var s = '';
    for (var i = 0; i < ABL.length; i++) {
      var a = ABL[i];
      s += '<div class="vla-acard' + (i === ABL.length - 1 ? ' vla-acard-hi' : '') + '">' +
        '<div class="vla-aname">' + a.name + '</div>' +
        '<div class="vla-amap">' + a.map + ' <small>mAP50</small></div>' +
        '<div class="vla-asub">' + a.sub + '</div>' +
        (a.tag ? '<span class="vla-atag">' + a.tag + '</span>' : '') +
        '</div>';
    }
    return s;
  }
  function promptChips() {
    var s = '';
    for (var i = 0; i < PROMPTS.length; i++) s += '<span class="vla-pchip">"' + PROMPTS[i] + '"</span>';
    return s;
  }

  var HTML =
    '<div class="vla-wrap">' +
      '<div class="vla-head">🎮 互動走讀|對不齊就融不好:AC-CA 對齊 + VL-CAT 訓練期蒸餾</div>' +
      '<div class="vla-body">' +

        // ---- §1 空間對齊 ----
        '<div class="vla-sec">' +
          '<div class="vla-sectitle">1|空間對齊實驗室(AC-CA 的 SFS)—— 把跑掉的 IR 拉回來</div>' +
          '<div class="vla-qrow"><span class="vla-qlabel">融合品質(示意)</span>' +
            '<div class="vla-track"><div class="vla-fill vla-qfill"></div></div>' +
            '<span class="vla-qval vla-qv" role="status"></span></div>' +
          '<canvas class="vla-cv vla-cv1" width="480" height="270" role="img" ' +
            'aria-label="夜間街景示意:RGB 層行人輪廓暗淡,IR 層熱斑半透明疊加;拖曳滑桿可讓 IR 層偏移產生鬼影"></canvas>' +
          '<div class="vla-legend"><span class="vla-mk-rgb">■</span> RGB 層:弱光下輪廓暗淡・' +
            '<span class="vla-mk-ir">●</span> IR 層:熱斑半透明疊加(虛線=IR 認為行人的位置)—— 拖滑桿模擬雙感測器未對齊</div>' +
          '<div class="vla-controls">' +
            '<label class="vla-ctl">IR 水平偏移<input type="range" class="vla-ox" min="-20" max="20" step="1" value="12" aria-label="IR 層水平偏移,正負 20 像素">' +
              '<span class="vla-val vla-oxval"></span></label>' +
            '<label class="vla-ctl">IR 垂直偏移<input type="range" class="vla-oy" min="-20" max="20" step="1" value="-8" aria-label="IR 層垂直偏移,正負 20 像素">' +
              '<span class="vla-val vla-oyval"></span></label>' +
            '<button type="button" class="vla-btn vla-btn-acc vla-align">⚡ SFS 對齊</button>' +
          '</div>' +
          '<div class="vla-blocks">' +
            '<div class="vla-block"><b>① SFS 相似特徵選擇</b>偏移網路 D 預測像素位移 f<sub>o</sub>=Tanh(D(Concat(f<sub>v</sub>,f<sub>t</sub>))),配參考網格(下採樣率 s=12)+ <b>bicubic 重採樣</b>,把兩模態顯式空間對齊。</div>' +
            '<div class="vla-block"><b>② CFE 上下文特徵抽取</b>先做<b>模態正規化</b>(減均值、除標準差)對齊兩模態的特徵分布,再經卷積投影成融合用的 Q/K/V。</div>' +
            '<div class="vla-block"><b>③ CAF 跨模態注意力融合</b>M=softmax(Q·K<sup>T</sup>/√d)——用一個模態的 query 比對另一模態的 key,輸出=自身正規化特徵+conv(M·V)。</div>' +
          '</div>' +
          '<p class="vla-fact">📉 <b>融合反而更省</b>:AC-CA 取代標準 transformer 融合後,精度上升之外,' +
            '<b>參數 44.5M→34.7M、FPS 47.1→52</b>(消融 Table 3)—— SFS 先對齊、去冗餘的功勞。</p>' +
        '</div>' +

        // ---- §2 惡劣天候切換 ----
        '<div class="vla-sec">' +
          '<div class="vla-sectitle">2|惡劣天候切換 —— RGB 隨情境崩、IR 穩</div>' +
          '<div class="vla-scen" role="group" aria-label="切換天候情境">' +
            '<button type="button" class="vla-scbtn" data-vla-w="day">☀️ 白天</button>' +
            '<button type="button" class="vla-scbtn vla-on" data-vla-w="night">🌙 夜晚</button>' +
            '<button type="button" class="vla-scbtn" data-vla-w="rainfog">🌧 雨霧</button>' +
          '</div>' +
          '<canvas class="vla-cv vla-cv2" width="480" height="270" role="img" aria-label=""></canvas>' +
          '<div class="vla-mbars">' +
            '<div class="vla-qrow"><span class="vla-qlabel">RGB 可用度(示意)</span>' +
              '<div class="vla-track"><div class="vla-fill vla-rgbfill" style="background:var(--accent,#0969da)"></div></div>' +
              '<span class="vla-qval vla-rgbv" style="color:var(--accent,#0969da)"></span></div>' +
            '<div class="vla-qrow"><span class="vla-qlabel">IR 可用度(示意)</span>' +
              '<div class="vla-track"><div class="vla-fill vla-irfill" style="background:var(--orange,#ec7500)"></div></div>' +
              '<span class="vla-qval vla-irv" style="color:var(--orange,#ec7500)"></span></div>' +
          '</div>' +
          '<p class="vla-caption"></p>' +
          '<p class="vla-fact">🗂 這正是<b>自建資料集</b>的設計:<b>43,420</b> 對 RGB-IR 影像、<b>132,905</b> 個標註,' +
            '640×480、<b>空間對齊+時間同步</b>,涵蓋<b>日/夜+雨霧</b>(行人/汽車/機車)—— 比 M³FD 更廣更難。</p>' +
          '<div class="vla-cards">' +
            '<span class="vla-chip">M³FD All mAP50 <b>86.67%</b></span>' +
            '<span class="vla-chip vla-chip-g">+6.17% vs TarDAL(early fusion)</span>' +
            '<span class="vla-chip vla-chip-g">+4.78% vs QFDet(late fusion)</span>' +
            '<span class="vla-chip">Adverse 子集 84.58% 最高</span>' +
            '<span class="vla-chip">自建 All 79.42(+2.17 vs 第二名 ICAFusion)</span>' +
          '</div>' +
          '<div class="vla-note">⚠️ 另一面:論文也提醒 <b>IR 並非永遠可靠</b>(前景/背景溫差小時同樣沒資訊)——' +
            '所以需要第 3 節的 VL-CAT 動態判斷「此刻哪個模態可信」,而不是只看照度。</div>' +
        '</div>' +

        // ---- §3 VL-CAT ----
        '<div class="vla-sec">' +
          '<div class="vla-sectitle">3|VL-CAT:CLIP 講師只在訓練時到場</div>' +
          '<div class="vla-controls vla-modebar">' +
            '<button type="button" class="vla-btn vla-mode" aria-pressed="false">🏋️ 訓練 ⇄ 🚀 推論</button>' +
            '<span class="vla-modelabel" role="status"></span>' +
          '</div>' +
          '<div class="vla-svgwrap">' + SVG + '</div>' +
          '<div class="vla-legend">實線=推論也在;<span class="vla-mk-ir">橘色虛線=只在訓練期存在</span>:' +
            'CLIP 語意經 L2 蒸餾(L<sub>T</sub>、L<sub>I</sub>)注入通道注意力。AC-CA 與 VL-CAT 都接在雙流骨幹的最後三層。</div>' +
          '<p class="vla-modetext"></p>' +
          '<div class="vla-prompts"><b>4 個模態品質 prompt(逐字)</b>:把「模態此刻好不好」變成 CLIP 可打分的語意<br>' + promptChips() + '</div>' +
          '<div class="vla-abl">' + ablCards() + '</div>' +
          '<div class="vla-legend">消融(Table 3,自建資料集)。VL-CAT 對 <b>mAP75 提升最大(40.77→46.14,+5.37)</b>→ 語意蒸餾主要改善定位精度;' +
            'Thermal-only 最差 → 單靠 IR 不夠,要融合。</div>' +
        '</div>' +

        '<div class="vla-src">⚠️ 本互動為概念示意:街景、融合品質條與模態可用度條皆<b>非真實影像處理或量測</b>。' +
          '數字出處:M³FD 對比=Table 1、自建資料集=Table 2、消融=Table 3(詳見本頁〈VL-ACFDet〉走讀內文 vlacfdet.md)。</div>' +
      '</div>' +
    '</div>';

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  // ---- 決定性偽隨機(雨紋/雜訊/亮窗固定圖樣,重繪不閃爍)----
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function rr(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function irBlob(ctx, x, y, r, intensity, ysc) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(1, ysc || 1);
    var g = ctx.createRadialGradient(0, 0, r * 0.08, 0, 0, r);
    g.addColorStop(0, 'rgba(255,251,235,' + (0.95 * intensity) + ')');
    g.addColorStop(0.35, 'rgba(255,201,99,' + (0.85 * intensity) + ')');
    g.addColorStop(0.72, 'rgba(236,117,0,' + (0.5 * intensity) + ')');
    g.addColorStop(1, 'rgba(233,49,71,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // ---- 場景繪製(mode:'day'|'night'|'rainfog';ox,oy = IR 層偏移)----
  function drawScene(ctx, mode, ox, oy) {
    var W = CW, H = CH;
    var night = mode === 'night', day = mode === 'day', fog = mode === 'rainfog';

    // 天空
    var sky = ctx.createLinearGradient(0, 0, 0, HZ);
    if (day) { sky.addColorStop(0, '#8fc1ec'); sky.addColorStop(1, '#dcecf8'); }
    else if (fog) { sky.addColorStop(0, '#3a434f'); sky.addColorStop(1, '#5b6570'); }
    else { sky.addColorStop(0, '#070c17'); sky.addColorStop(1, '#13203a'); }
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, HZ);
    if (day) { ctx.fillStyle = '#ffd23f'; ctx.beginPath(); ctx.arc(432, 40, 16, 0, Math.PI * 2); ctx.fill(); }
    if (night) {
      ctx.fillStyle = '#e8e4cf'; ctx.beginPath(); ctx.arc(432, 38, 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#0d1526'; ctx.beginPath(); ctx.arc(426, 34, 11, 0, Math.PI * 2); ctx.fill();
    }

    // 建築(幾何示意)
    var bc = day ? '#9fb0c6' : (fog ? '#4a5460' : '#141b2b');
    ctx.fillStyle = bc;
    ctx.fillRect(0, HZ - 96, 118, 96);
    ctx.fillRect(126, HZ - 64, 66, 64);
    ctx.fillRect(338, HZ - 112, 142, 112);
    // 窗
    var wr = mulberry32(7), wx, wy;
    for (wx = 10; wx <= 98; wx += 22) {
      for (wy = HZ - 86; wy <= HZ - 32; wy += 18) {
        var lit1 = wr() < 0.45;
        ctx.fillStyle = day ? '#e6eef7' : (fog ? 'rgba(220,228,238,.15)' : (lit1 ? 'rgba(255,216,112,.85)' : '#1d2740'));
        ctx.fillRect(wx, wy, 12, 10);
      }
    }
    for (wx = 348; wx <= 468; wx += 24) {
      for (wy = HZ - 102; wy <= HZ - 22; wy += 20) {
        var lit2 = wr() < 0.4;
        ctx.fillStyle = day ? '#e6eef7' : (fog ? 'rgba(220,228,238,.15)' : (lit2 ? 'rgba(255,216,112,.8)' : '#1d2740'));
        ctx.fillRect(wx, wy, 13, 11);
      }
    }

    // 路面
    var rd = ctx.createLinearGradient(0, HZ, 0, H);
    if (day) { rd.addColorStop(0, '#7d8794'); rd.addColorStop(1, '#5f6874'); }
    else if (fog) { rd.addColorStop(0, '#39404b'); rd.addColorStop(1, '#262c35'); }
    else { rd.addColorStop(0, '#151a24'); rd.addColorStop(1, '#0e1219'); }
    ctx.fillStyle = rd;
    ctx.fillRect(0, HZ, W, H - HZ);
    // 車道虛線
    ctx.strokeStyle = day ? 'rgba(255,255,255,.9)' : (fog ? 'rgba(190,205,235,.18)' : 'rgba(190,205,235,.32)');
    ctx.lineWidth = 4;
    ctx.setLineDash([28, 30]);
    ctx.beginPath();
    ctx.moveTo(8, 222);
    ctx.lineTo(W, 222);
    ctx.stroke();
    ctx.setLineDash([]);

    // 路燈
    ctx.fillStyle = day ? '#6a7482' : '#39414f';
    ctx.fillRect(168, 88, 4, 112);
    ctx.fillRect(168, 84, 38, 4);
    ctx.beginPath(); ctx.arc(206, 94, 5, 0, Math.PI * 2); ctx.fill();
    if (night) {
      var cone = ctx.createLinearGradient(0, 96, 0, 208);
      cone.addColorStop(0, 'rgba(255,220,130,.30)');
      cone.addColorStop(1, 'rgba(255,220,130,0)');
      ctx.fillStyle = cone;
      ctx.beginPath(); ctx.moveTo(206, 96); ctx.lineTo(166, 208); ctx.lineTo(246, 208); ctx.closePath(); ctx.fill();
      ctx.fillStyle = 'rgba(255,236,170,.9)'; ctx.beginPath(); ctx.arc(206, 94, 3, 0, Math.PI * 2); ctx.fill();
    }
    if (fog) { // 霧中燈暈
      var halo = ctx.createRadialGradient(206, 94, 2, 206, 94, 46);
      halo.addColorStop(0, 'rgba(255,224,150,.22)');
      halo.addColorStop(1, 'rgba(255,224,150,0)');
      ctx.fillStyle = halo;
      ctx.beginPath(); ctx.arc(206, 94, 46, 0, Math.PI * 2); ctx.fill();
    }

    // 車(車頭朝左)
    ctx.fillStyle = day ? '#4a6584' : (fog ? '#333c49' : '#1c2534');
    rr(ctx, 298, 170, 116, 26, 8); ctx.fill();
    rr(ctx, 320, 152, 60, 22, 6); ctx.fill();
    ctx.fillStyle = day ? '#cfe0ee' : 'rgba(120,150,190,.25)';
    rr(ctx, 326, 156, 48, 14, 4); ctx.fill();
    ctx.fillStyle = day ? '#2c3540' : '#0b0f16';
    ctx.beginPath(); ctx.arc(324, 198, 9, 0, Math.PI * 2); ctx.arc(390, 198, 9, 0, Math.PI * 2); ctx.fill();
    if (night || fog) {
      ctx.fillStyle = 'rgba(255,240,180,.85)';
      ctx.fillRect(298, 176, 4, 6);
      var beam = ctx.createLinearGradient(298, 0, 250, 0);
      beam.addColorStop(0, 'rgba(255,240,180,' + (fog ? '.10' : '.18') + ')');
      beam.addColorStop(1, 'rgba(255,240,180,0)');
      ctx.fillStyle = beam;
      ctx.beginPath(); ctx.moveTo(298, 176); ctx.lineTo(250, 170); ctx.lineTo(250, 194); ctx.lineTo(298, 184); ctx.closePath(); ctx.fill();
    }

    // 行人 —— RGB 層(隨情境退化)
    var headC, bodyC, limbC;
    if (day) { headC = '#e8c19b'; bodyC = '#2563c9'; limbC = '#3a4453'; }
    else if (fog) { headC = 'rgba(150,158,170,.30)'; bodyC = 'rgba(135,146,160,.30)'; limbC = 'rgba(120,130,145,.28)'; }
    else { headC = 'rgba(150,160,180,.55)'; bodyC = 'rgba(118,132,158,.5)'; limbC = 'rgba(100,114,140,.48)'; }
    ctx.fillStyle = limbC;
    rr(ctx, 214, 187, 7, 24, 3); ctx.fill();
    rr(ctx, 223, 187, 7, 24, 3); ctx.fill();
    rr(ctx, 207, 158, 6, 22, 3); ctx.fill();
    rr(ctx, 231, 158, 6, 22, 3); ctx.fill();
    ctx.fillStyle = bodyC;
    rr(ctx, 211, 155, 22, 32, 7); ctx.fill();
    ctx.fillStyle = headC;
    ctx.beginPath(); ctx.arc(222, 146, 7, 0, Math.PI * 2); ctx.fill();

    // 天候覆蓋(只影響 RGB 層,畫在 IR 之前)
    if (fog) {
      var rn = mulberry32(11), i;
      ctx.strokeStyle = 'rgba(186,202,224,.30)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (i = 0; i < 85; i++) {
        var rx0 = rn() * W, ry0 = rn() * (H - 10);
        ctx.moveTo(rx0, ry0);
        ctx.lineTo(rx0 - 7, ry0 + 16);
      }
      ctx.stroke();
      for (i = 0; i < 150; i++) {
        ctx.fillStyle = (i % 2) ? 'rgba(255,255,255,.09)' : 'rgba(0,0,0,.18)';
        ctx.fillRect(rn() * W, rn() * H, 2, 2);
      }
      ctx.fillStyle = 'rgba(196,205,216,.20)';
      ctx.fillRect(0, 0, W, H);
      var band = ctx.createLinearGradient(0, 100, 0, 214);
      band.addColorStop(0, 'rgba(205,212,222,0)');
      band.addColorStop(0.5, 'rgba(205,212,222,.34)');
      band.addColorStop(1, 'rgba(205,212,222,0)');
      ctx.fillStyle = band;
      ctx.fillRect(0, 100, W, 114);
    }
    if (night) {
      var vig = ctx.createRadialGradient(240, 135, 130, 240, 135, 330);
      vig.addColorStop(0, 'rgba(0,0,0,0)');
      vig.addColorStop(1, 'rgba(0,0,0,.42)');
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, W, H);
    }

    // IR 層(熱斑;不受天候影響,只受 ox/oy 偏移)
    irBlob(ctx, 222 + ox, 146 + oy, 13, 1);          // 行人頭
    irBlob(ctx, 222 + ox, 178 + oy, 17, 1, 1.9);     // 行人軀幹+腿
    irBlob(ctx, 312 + ox, 182 + oy, 12, 0.45);       // 車引擎微熱
    ctx.save();
    ctx.strokeStyle = 'rgba(255,205,130,.8)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.ellipse(222 + ox, 172 + oy, 20, 42, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // ---- 主入口 ----
  window.initVlacfdetDemo = function (container) {
    if (!container) return;
    if (typeof container.__vlaCleanup === 'function') container.__vlaCleanup();
    injectStyle();
    container.innerHTML = HTML;

    function $(sel) { return container.querySelector(sel); }

    var cv1 = $('.vla-cv1'), cv2 = $('.vla-cv2');
    var ctx1 = cv1.getContext('2d'), ctx2 = cv2.getContext('2d');
    var qFill = $('.vla-qfill'), qVal = $('.vla-qv');
    var oxIn = $('.vla-ox'), oyIn = $('.vla-oy');
    var oxVal = $('.vla-oxval'), oyVal = $('.vla-oyval');
    var alignBtn = $('.vla-align');
    var rgbFill = $('.vla-rgbfill'), rgbV = $('.vla-rgbv');
    var irFill = $('.vla-irfill'), irV = $('.vla-irv');
    var caption = $('.vla-caption');
    var modeBtn = $('.vla-mode'), modeLabel = $('.vla-modelabel'), modeText = $('.vla-modetext');
    var svgWrap = $('.vla-svgwrap');

    var st = { ox: 12, oy: -8, weather: 'night', train: true, rafId: 0, destroyed: false };

    // prefers-reduced-motion:動畫改瞬間到位
    var mq = null;
    try { mq = window.matchMedia('(prefers-reduced-motion: reduce)'); } catch (e) { mq = null; }
    function reduced() { return !!(mq && mq.matches); }
    function onMqChange() {
      if (reduced() && st.rafId) { // 對齊動畫進行中 → 直接到位
        cancelAnimationFrame(st.rafId);
        st.rafId = 0;
        setOffset(0, 0);
      }
    }
    if (mq) {
      if (mq.addEventListener) mq.addEventListener('change', onMqChange);
      else if (mq.addListener) mq.addListener(onMqChange);
    }

    function fmtPx(v) { return (v > 0 ? '+' : '') + v + 'px'; }

    function updateQuality() {
      var d = Math.hypot(st.ox, st.oy);
      var q = Math.max(0, 1 - d / MAX_DIST); // 偏移距離 → 品質,線性映射(示意)
      var pct = Math.round(q * 100);
      qFill.style.width = pct + '%';
      qFill.style.background = q >= 0.7 ? C_GREEN : (q >= 0.4 ? C_ORANGE : C_RED);
      qVal.style.color = q >= 0.7 ? C_GREEN : (q >= 0.4 ? C_ORANGE : C_RED);
      qVal.textContent = pct + '%';
    }

    function setOffset(x, y) {
      st.ox = x; st.oy = y;
      oxIn.value = String(Math.round(x));
      oyIn.value = String(Math.round(y));
      oxVal.textContent = fmtPx(Math.round(x));
      oyVal.textContent = fmtPx(Math.round(y));
      drawScene(ctx1, 'night', x, y);
      updateQuality();
    }

    function cancelAlign() {
      if (st.rafId) { cancelAnimationFrame(st.rafId); st.rafId = 0; }
    }

    // ⚡ SFS 對齊:easeInOutCubic 滑回 (0,0);reduced-motion 或分頁隱藏(rAF 停擺)時瞬間到位
    function alignSFS() {
      cancelAlign();
      if (reduced() || document.hidden) { setOffset(0, 0); return; }
      var sx = st.ox, sy = st.oy;
      if (sx === 0 && sy === 0) return;
      var t0 = performance.now(), dur = 750;
      function step(now) {
        if (st.destroyed) return;
        var t = Math.min(1, (now - t0) / dur);
        var e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        setOffset(sx * (1 - e), sy * (1 - e));
        if (t < 1) { st.rafId = requestAnimationFrame(step); }
        else { st.rafId = 0; setOffset(0, 0); }
      }
      st.rafId = requestAnimationFrame(step);
    }

    function onSlider() {
      cancelAlign(); // 使用者接手 → 中斷對齊動畫
      setOffset(parseInt(oxIn.value, 10) || 0, parseInt(oyIn.value, 10) || 0);
    }

    // §2 天候切換
    function setWeather(w) {
      st.weather = w;
      var conf = WEATHERS[w];
      var btns = container.querySelectorAll('.vla-scbtn');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle('vla-on', btns[i].getAttribute('data-vla-w') === w);
      }
      drawScene(ctx2, w, 0, 0); // §2 已對齊,只看天候;IR 熱斑始終穩定
      rgbFill.style.width = conf.rgb + '%';
      rgbV.textContent = conf.rgb + '%';
      irFill.style.width = conf.ir + '%';
      irV.textContent = conf.ir + '%';
      caption.innerHTML = conf.cap;
      cv2.setAttribute('aria-label', conf.aria);
    }

    // §3 訓練 ⇄ 推論
    function setTrain(train) {
      st.train = train;
      svgWrap.classList.toggle('vla-inf', !train);
      modeBtn.setAttribute('aria-pressed', String(!train));
      modeLabel.textContent = train
        ? '目前:🏋️ 訓練期 —— CLIP 老師在場'
        : '目前:🚀 推論期 —— CLIP 已退場';
      modeText.innerHTML = train ? TRAIN_TXT : INFER_TXT;
    }

    // ---- 綁事件 ----
    oxIn.addEventListener('input', onSlider);
    oyIn.addEventListener('input', onSlider);
    alignBtn.addEventListener('click', alignSFS);
    var scBtns = container.querySelectorAll('.vla-scbtn');
    function onScen(ev) { setWeather(ev.currentTarget.getAttribute('data-vla-w')); }
    for (var i = 0; i < scBtns.length; i++) scBtns[i].addEventListener('click', onScen);
    modeBtn.addEventListener('click', function () { setTrain(!st.train); });

    // ---- 初始畫面 ----
    setOffset(st.ox, st.oy);
    setWeather('night');
    setTrain(true);

    // ---- 清理(同容器重複 init 時自動呼叫) ----
    container.__vlaCleanup = function () {
      st.destroyed = true;
      cancelAlign();
      if (mq) {
        if (mq.removeEventListener) mq.removeEventListener('change', onMqChange);
        else if (mq.removeListener) mq.removeListener(onMqChange);
      }
      container.innerHTML = ''; // 容器內元素連同 listener 一併移除
      container.__vlaCleanup = null;
    };
  };
})();
