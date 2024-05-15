---
sidebar_position: 5
---

# 5. Find a Remote Branch and Fetch it to Local with Git

`git branch -r` can list all the remote branch. In any large scale application code base, you are likely to get heaps of branch names. By piping grep, we can narrow down the remote branch

```bash
git branch -r | grep MDH-112
```

This gives us a fewer options.

```bash
  origin/MDH-112-dev-branch
  origin/MDH-112-experiment
  origin/MDH-112-delete
```

Then, fetch the branch from the remote

```bash
git fetch origin MDH-112-dev-branch
```

(2020-02-28)
