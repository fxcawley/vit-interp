// node arx.js "<arxiv query>" [max] [sortBy: relevance|submittedDate]
const { execFileSync } = require('child_process');
const q = process.argv[2]; const max = process.argv[3] || 15; const sort = process.argv[4] || 'relevance';
const url = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(q)}&max_results=${max}&sortBy=${sort}&sortOrder=descending`;
const t = execFileSync('curl', ['-s', '-m', '60', url], { encoding: 'utf8', maxBuffer: 50e6 });
const tot = (t.match(/totalResults>(\d+)</) || [])[1]; console.log(`totalResults=${tot}`);
function tag(e, name) { const re = new RegExp('<' + name + '(?:\\s[^>]*)?>([\\s\\S]*?)</' + name + '>'); const m = e.match(re); return m ? m[1].replace(/\s+/g, ' ').trim() : ''; }
for (const e of t.split('<entry>').slice(1)) {
  const abs = tag(e, 'summary');
  console.log(`\n[${tag(e, 'id').replace('http://arxiv.org/abs/', '')}] ${tag(e, 'published').slice(0, 10)}  ${tag(e, 'title')}  {${tag(e, 'arxiv:comment').slice(0, 120)}}\n   ${abs.slice(0, 420)}${abs.length > 420 ? '...' : ''}`);
}
