---
sidebar_position: 3
---

# 3. How to Rename a Branch with Git

Renaming a local branch is easy by using git branch command option -m. For the remote branch, we can create a new remote branch by pushing the renamed branch and then deleting the old branch.

Here are the steps.

(1) checkout

```bash
git checkout old-branch-name
```

(2) rename

```bash
git branch -m new-branch-name
```

(3) push the new branch

```bash
git push origin -u new-branch-name
```

(4) delete the old one

```bash
git push origin --delete old-branch-name
```

(2020-02-28)
