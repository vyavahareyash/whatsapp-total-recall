---
name: release-check
description: Comprehensive pre-flight audit, documentation synchronization, and release pipeline orchestration for WhatsApp Total Recall. Use before cutting any release, tagging, or updating PyPI.
---

# Release Check & Documentation Sync Skill

Use this skill whenever preparing a new release, tagging a version, bumping semver, or auditing documentation and distribution packages.

---

## 1. Automated Pre-Flight Audit

Run the pre-flight verification script:
```bash
python skills/release-check/scripts/check_release.py
```
This automatically verifies:
1. Version alignment across `pyproject.toml`, `apps/WhatsAppTotalRecallBar/Info.plist`, and `whatsapp_total_recall/mcp.py`.
2. Package README pointer: `pyproject.toml` `readme = "README.pypi.md"`.
3. PyPI README hygiene: zero relative links (which 404 on PyPI) and inclusion in sdist targets.
4. PyPI Trusted Publishing alignment: `.github/workflows/release.yml` with `environment: pypi` and `id-token: write`.

---

## 2. Documentation Update Overhead Workflow

Every release must audit and synchronize documentation across two distinct scopes:

### A. Scoped PyPI README (`README.pypi.md`)
PyPI should reflect only the Python package, CLI, and MCP server:
- **Zero Relative Links**: All links to code, docs, or issues must use absolute GitHub URLs (`https://github.com/vyavahareyash/whatsapp-total-recall/...`).
- **CLI Command Syntax**: Verify `whatsapp-total-recall` command and new subcommands (e.g. `vault approve --all`, `vault revoke --all`) are documented.
- **MCP Client Blocks**: Verify JSON snippet blocks for:
  - Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)
  - Cursor / Windsurf (`.cursor/mcp.json`)
  - Claude Code CLI (`claude mcp add ...`)
- **Native App Link**: Direct users looking for the Menu Bar app to the GitHub Releases `.dmg` download.

### B. Full GitHub README (`README.md`)
GitHub README covers the entire monorepo:
- macOS Menu Bar app architecture and screenshots.
- Native build instructions (`make build`, `make dmg`).
- Local RAG and DAG attribution design.

### C. Release Notes Generation
Extract changes since the last release tag:
```bash
PREV_TAG=$(git describe --tags --abbrev=0 2>/dev/null || git rev-list --max-parents=0 HEAD)
git log ${PREV_TAG}..HEAD --oneline --no-merges
```
Categorize changes into:
- **Features / MCP Tools**: Newly exposed tools, CLI flags, or capabilities.
- **Fixes**: Bug fixes, sandbox permission fixes, or edge-case handling.
- **Documentation**: New setup guides, MCP snippets, or PyPI assets.

---

## 3. Local Quality & Build Verification

Before committing release changes:
```bash
# 1. Lint & Format check
ruff check .
ruff format --check .

# 2. Pytest suite
pytest -v

# 3. Clean package build
uv build  # or python -m build
```

---

## 4. Branching & PR Procedure (Repo Rule GH013)

Direct pushes to `main` are rejected by repository branch protection. All changes must be merged via PR:

1. **Create release branch**:
   ```bash
   git checkout -b release/v<VERSION>
   ```
2. **Commit changes**:
   ```bash
   git commit -m "chore(release): prepare v<VERSION>"
   ```
3. **Push & Open PR**:
   - Push branch to origin.
   - Open Pull Request to `main`.
   - Merge Pull Request into `main`.

---

## 5. Tagging & Automated Dispatch

Once merged into `main`:
```bash
git checkout main
git pull origin main
git tag -a v<VERSION> -m "Release v<VERSION>"
git push origin v<VERSION>
```

Pusing the `v*` tag triggers `.github/workflows/release.yml`:
- Runs tests on Ubuntu & macOS.
- Compiles universal macOS `.dmg`.
- Publishes GitHub Release with checksums.
- Publishes package to PyPI via Trusted Publishing OIDC.

---

## 6. Post-Release Verification

1. Inspect GitHub Actions run: `https://github.com/vyavahareyash/whatsapp-total-recall/actions`.
2. Inspect PyPI release: `https://pypi.org/project/whatsapp-total-recall/<VERSION>/`.
3. Smoke test package via uvx:
   ```bash
   uvx --from "whatsapp-total-recall==<VERSION>[vec]" whatsapp-total-recall --help
   ```
