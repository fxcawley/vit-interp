// node gs.js "<query>" [ylo] -> Google Scholar results (title, url, snippet). Use sparingly.
const { execFileSync } = require('child_process');
const q = process.argv[2]; const ylo = process.argv[3] || '2024';
const url = `https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=${encodeURIComponent(q)}&as_ylo=${ylo}`;
const h = execFileSync('curl', ['-sL', '-m', '60', '-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36', '-H', 'Accept-Language: en-US,en;q=0.9', url], { encoding: 'utf8', maxBuffer: 50e6 });
const strip = s => s.replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&hellip;/g, '...').replace(/\s+/g, ' ').trim();
const items = h.split('<div class="gs_r gs_or gs_scl"').slice(1);
if (!items.length) { console.log('(no results; raw length ' + h.length + ') ' + strip(h).slice(0, 200)); process.exit(0); }
items.forEach((it, i) => {
  const t = it.match(/<h3 class="gs_rt"[^>]*>([\s\S]*?)<\/h3>/); const a = t ? t[1].match(/href="([^"]+)"/) : null;
  const meta = it.match(/<div class="gs_a"[^>]*>([\s\S]*?)<\/div>/); const s = it.match(/<div class="gs_rs"[^>]*>([\s\S]*?)<\/div>/);
  const cit = it.match(/Cited by (\d+)/);
  console.log(`\n${i + 1}. ${t ? strip(t[1]) : ''}  ${cit ? '[cited by ' + cit[1] + ']' : ''}\n   ${a ? a[1] : ''}\n   ${meta ? strip(meta[1]) : ''}\n   ${s ? strip(s[1]).slice(0, 300) : ''}`);
});
