# Project rules

## Git

- Commit messages carry no signing lines or trailers: no "Generated with ...", no "Co-Authored-By", no tool attribution. Subject line plus an optional short body about why.
- Do not rewrite history (amend, rebase, reset of pushed or earlier commits) without asking first.

## Research hygiene

- Before building on an idea, run the `lit-check` skill (scoop check, primary-source verification, idea-sink guard). Mark facts `[verified YYYY-MM-DD]` only after reading the primary source.
- Preregister predictions and kill criteria before running experiments; exploratory results are labeled exploratory.
- No KLA data in anything intended for publication.

## Environment notes

- Windows host has no Python on PATH. Use WSL Ubuntu (`wsl -d Ubuntu -- bash -lc '...'`): Python 3.12, PyTorch 2.10 + CUDA, RTX PRO 2000 (8 GB), 16 cores, 31 GB RAM.
- Network goes through a Zscaler proxy. The `brave-search` MCP tool works; the `fetch` MCP tool does not; `curl` works for arXiv, OpenAlex, Google Scholar, GitHub pages and Zenodo. Raw code files from GitHub are quarantined (use `git clone` or the GitHub API tree). Some blog hosts are blocked outright.
- Shell heredocs lose one level of backslash escaping; write scripts with the file-write tool, not `cat <<EOF`. The write tool resolves `/tmp` to `C:\tmp`, not the Git Bash temp directory.
