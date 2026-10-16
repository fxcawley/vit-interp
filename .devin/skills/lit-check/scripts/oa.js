// node oa.js "<query>" [n] [fromYear] -> OpenAlex works search (title, year, venue, cites, abstract snippet)
const { execFileSync } = require('child_process');
const q = process.argv[2]; const n = process.argv[3] || 10; const from = process.argv[4] || '2023';
const url = `https://api.openalex.org/works?search=${encodeURIComponent(q)}&per-page=${n}&filter=from_publication_date:${from}-01-01&sort=relevance_score:desc&mailto=litreview@example.com`;
const t = execFileSync('curl', ['-s', '-m', '60', url], { encoding: 'utf8', maxBuffer: 50e6 });
let j; try { j = JSON.parse(t); } catch { console.log(t.slice(0, 300)); process.exit(0); }
console.log(`count=${j.meta?.count}`);
for (const w of (j.results || [])) {
  let abs = '';
  if (w.abstract_inverted_index) { const arr = []; for (const [word, pos] of Object.entries(w.abstract_inverted_index)) for (const p of pos) arr[p] = word; abs = arr.join(' '); }
  const src = w.primary_location?.source?.display_name || '-';
  const arx = (w.locations || []).map(l => l.landing_page_url || '').find(u => u.includes('arxiv')) || '';
  console.log(`\n${w.publication_year} cites=${w.cited_by_count} [${src}] ${arx}\n  ${w.display_name}\n   ${abs.slice(0, 350)}${abs.length > 350 ? '...' : ''}`);
}
