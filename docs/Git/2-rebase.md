---
sidebar_position: 2
---

# 2. How to Rebase with Git

Here are the steps to rebase from master with Git. After doing this a few times, rebase is not so scary any more. Rebase vs Merge can be contentious. Generally speaking, I prefer to rebase because it creates a cleaner history.

(1) Make sure to be in the feature branch

```bash
git pull origin master --rebase
```

(2) If there are merge conflicts, you need to resolve it. Then, add the change (you do not need to do git commit).

```bash
git add *
git rebase --continue
```

(3) We need to force push because rebasing rewrite the history by creating a new tree.

```bash
git push origin -f
```

(2020-02-28)
