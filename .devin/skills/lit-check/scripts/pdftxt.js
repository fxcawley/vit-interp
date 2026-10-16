// node pdftxt.js <file.pdf> [grep-regex] [ctx] -> crude text extraction (inflate streams, pull Tj/TJ strings)
const fs = require('fs'); const zlib = require('zlib');
const [file, pat, ctx = '200'] = process.argv.slice(2);
const buf = fs.readFileSync(file);
const s = buf.toString('latin1');
let out = [];
let idx = 0;
while (true) {
  const a = s.indexOf('stream', idx); if (a < 0) break;
  let b = a + 6; if (s[b] === '\r') b++; if (s[b] === '\n') b++;
  const e = s.indexOf('endstream', b); if (e < 0) break;
  const raw = buf.subarray(b, e);
  let txt = null;
  try { txt = zlib.inflateSync(raw).toString('latin1'); } catch { try { txt = zlib.inflateSync(raw.subarray(0, raw.length - 1)).toString('latin1'); } catch { txt = null; } }
  if (txt && /T[Jj]/.test(txt)) {
    // extract strings inside ( ) within TJ arrays and Tj operators
    const parts = [];
    const re = /\[((?:[^\]\\]|\\.)*)\]\s*TJ|\(((?:[^)\\]|\\.)*)\)\s*Tj|T\*|Td|TD|'|"/g;
    let m;
    while ((m = re.exec(txt))) {
      if (m[1] !== undefined) {
        const strs = []; const re2 = /\(((?:[^)\\]|\\.)*)\)|(-?\d+\.?\d*)/g; let k;
        while ((k = re2.exec(m[1]))) { if (k[1] !== undefined) strs.push(k[1]); else if (parseFloat(k[2]) < -200) strs.push(' '); }
        parts.push(strs.join(''));
      } else if (m[2] !== undefined) parts.push(m[2]);
      else parts.push('\n');
    }
    out.push(parts.join('').replace(/\\\(/g, '(').replace(/\\\)/g, ')').replace(/\\\\/g, '\\'));
  }
  idx = e + 9;
}
let text = out.join('\n').replace(/\x0c/g, 'fi').replace(/\x0d/g, 'fl').replace(/[ \t]+/g, ' ').replace(/\n{2,}/g, '\n');
console.log(`(total ${text.length} chars)`);
if (pat) { const re = new RegExp(pat, 'ig'); let m; let n = 0; while ((m = re.exec(text)) && n < 40) { n++; console.log(`--- @${m.index}: ` + text.slice(Math.max(0, m.index - +ctx), m.index + +ctx).replace(/\n/g, ' ')); } }
else console.log(text.slice(0, 6000));
