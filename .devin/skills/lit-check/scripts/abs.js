// node abs.js <id> [<id>...]  -> full abstract(s)
const { execFileSync } = require('child_process');
const ids = process.argv.slice(2).join(',');
const t = execFileSync('curl', ['-s', '-m', '60', `https://export.arxiv.org/api/query?id_list=${ids}&max_results=50`], { encoding: 'utf8', maxBuffer: 50e6 });
function tag(e, name) { const re = new RegExp('<' + name + '(?:\\s[^>]*)?>([\\s\\S]*?)</' + name + '>'); const m = e.match(re); return m ? m[1].replace(/\s+/g, ' ').trim() : ''; }
for (const e of t.split('<entry>').slice(1)) console.log(`\n[${tag(e, 'id').replace('http://arxiv.org/abs/', '')}] ${tag(e, 'published').slice(0, 10)} (updated ${tag(e, 'updated').slice(0, 10)})\nTITLE: ${tag(e, 'title')}\nCOMMENT: ${tag(e, 'arxiv:comment')}\nABSTRACT: ${tag(e, 'summary')}`);
