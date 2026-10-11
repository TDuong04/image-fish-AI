# The public copy of the repository

Made 2026-10-11 with the owner's approval, so that the repository can be published without a personal file and private identifiers that sit in this repository's history.
**This repository was not rewritten.** The public copy is a separate clone, rewritten with `git filter-repo`, and **nothing has been pushed**.

| | |
|---|---|
| Location (this workstation) | `Fish_Detection_public/`, next to this repository |
| What was removed | the personal file added in one early commit (from every commit); the commit message that described it was reworded |
| What was replaced, in every file and commit message | private hostnames and the team repository address (placeholders `<repository-url>`, `<owner>/<repository>`, `git.example.org`, `example.org`), the Unix account path of the server (`<repo>`), the name of another person's project (`another-project`), the Kaggle username (`your-kaggle-username`) |
| Tests | all pass in the public copy (327 at the time) |
| Commit hashes | every hash differs from this repository's; the old-to-new map is `docs/report/public_commit_map.txt`, so a `git_sha` in a run record can be translated |
| Author identity | unchanged: every commit carries the name and e-mail from the git configuration (see `docs/PUBLISH_CHECKLIST.md` item 5) |

## Regenerating it

The public copy is a snapshot: commits made here afterwards are not in it. Rebuild it just before publishing (needs `pip install git-filter-repo`):

```
git clone --no-local Fish_Detection Fish_Detection_public
cd Fish_Detection_public
git filter-repo --force --invert-paths --path <the personal file> --replace-text <rules> --replace-message <rules>
```

The rules file (`public_copy_exprs.txt`, one `literal:old==>new` line per replacement) is kept **outside both repositories**, next to them, because it names the very strings it removes; do not commit it.
After rebuilding, check the history is clean: `git log --all -p` must show no hostname, account path, project name or username from the rules file, and run the tests in the copy.
