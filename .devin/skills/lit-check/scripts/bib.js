// node bib.js <arxiv-id> [<arxiv-id> ...] -> BibTeX from arXiv's endpoint, concatenated to stdout
const { execFileSync } = require('child_process');
for (const id of process.argv.slice(2)) {
  try {
    const t = execFileSync('curl', ['-s', '-m', '30', `https://arxiv.org/bibtex/${id}`], { encoding: 'utf8' });
    process.stdout.write(t.trim() + '\n\n');
  } catch (e) { console.error(`failed: ${id}`); }
}
