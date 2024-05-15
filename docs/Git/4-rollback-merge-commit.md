---
sidebar_position: 4
---

# 4. How to Roll Back Merge Commit on Master Branch with Git

When rolling back a merge commit, I prefer to do it with git revert because it creates a new commit. Using -m 1 indicates that this is a merge and we want to roll back to the parent commit on the master branch. -m 2 is used to specify the dev branch.

```bash
git revert -m 1 merge-commit-sha
```

(2020-02-28)
