# GitHub Release Checklist

## 1) Create an empty GitHub repository

Create a new repository on GitHub and leave it empty (do **not** initialize with README).

Examples:
- `https://github.com/<USER>/<REPO>.git`
- `git@github.com:<USER>/<REPO>.git`

## 2) Commit curated release files

```bash
cd <AUTODL_ROOT>/ReferGaussian

git add \
  LICENSE \
  README.md \
  docs/index.html \
  docs/assets/githubio.css \
  docs/open_source_readiness_20260416.md \
  docs/reproducibility_20260416.md \
  docs/github_publish_steps.md \
  scripts/common.sh \
  scripts/setup_baseline_env.sh \
  scripts/setup_grounded_sam2.sh \
  scripts/setup_gsam2_env.sh \
  scripts/run_public_query_protocol.sh \
  scripts/list_public_protocol_queries.py \
  scripts/export_entitybank.sh \
  scripts/render_stellar_tube.sh \
  scripts/run_query_guided_full.sh

git commit -m "release: public ReferGaussian cleanup"
```

## 3) Push main branch

```bash
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin <YOUR_REPO_URL>
git push -u origin main
```

## 4) Enable GitHub Pages

- Open: `Settings -> Pages`
- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

Your page will be served at:

`https://<USER>.github.io/<REPO>/`

## 5) Replace placeholders before final announcement

In `docs/index.html`:
- `https://github.com/<ORG>/<REPO>`
- `https://<HG-LINK>`
- `https://arxiv.org/abs/<ARXIV-ID>`

In `README.md`:
- `https://huggingface.co/datasets/<ORG>/ReferGaussian-R4D-Bench-QA`
