// node txt.js <url> [start] [len] [grep-regex] -> fetch page, strip tags, print slice (or lines matching regex with context)
const { execFileSync } = require('child_process');
const [url, start = '0', len = '6000', pat] = process.argv.slice(2);
let h = execFileSync('curl', ['-sL', '-m', '120', '-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', url], { encoding: 'utf8', maxBuffer: 200e6 });
h = h.replace(/<script[\s\S]*?<\/script>/gi, '').replace(/<style[\s\S]*?<\/style>/gi, '').replace(/<[^>]+>/g, ' ')
  .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#39;|&rsquo;|&lsquo;/g, "'").replace(/&quot;|&ldquo;|&rdquo;/g, '"')
  .replace(/[ \t]+/g, ' ').replace(/\s*\n\s*/g, '\n').replace(/\n{2,}/g, '\n');
console.log(`(total ${h.length} chars)`);
if (pat) {
  const re = new RegExp(pat, 'i'); const lines = h.split('\n');
  lines.forEach((l, i) => { if (re.test(l)) console.log(`--- L${i}: ` + lines.slice(Math.max(0, i - 1), i + 2).join(' | ').slice(0, +len)); });
} else console.log(h.slice(+start, +start + +len));
