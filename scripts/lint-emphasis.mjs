// 用站上同版本 marked 渲染 papers/*.md 與 walkthrough.md,
// 檢查是否有粗體語法解析失敗(輸出殘留字面 **)。有 → exit 1。
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const { marked } = require('./vendor/marked.min.js');

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const files = [
  ...fs.readdirSync(path.join(root, 'papers')).filter(f => f.endsWith('.md')).map(f => 'papers/' + f),
  'walkthrough.md',
];

let bad = 0;
for (const f of files) {
  const md = fs.readFileSync(path.join(root, f), 'utf8');
  const html = marked.parse(md)
    .replace(/<code class="language-mermaid">[\s\S]*?<\/code>/g, '')
    .replace(/<code>[^<]*<\/code>/g, '');
  let i = -1;
  while ((i = html.indexOf('**', i + 1)) !== -1) {
    bad++;
    console.log(`⚠️ ${f} 粗體未解析: …${html.slice(Math.max(0, i - 50), i + 40).replace(/\n/g, '⏎')}…`);
  }
}
process.exit(bad ? 1 : 0);
