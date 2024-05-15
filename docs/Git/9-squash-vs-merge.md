---
sidebar_position: 9
---

# 9. Should We Use Squash and Merge Option?

When we merge PR into the base branch, we have an option to do Squash and Merge. My preference is to use it all the time. Let's talk about why this is the case.

First of all, what do we want from our commit history? We want to have a clean commit history with no noise. We also want to retain the history of change so that we can understand the change and context when we look back.

One commit should be sufficient for a lot of PRs. If you need to have multiple commits, the PR is too large and it should be broken down into smaller pieces.

However, there are many cases where you want to have a few commit histories so that we can trace the change and the PR history itself works as a documentation for the context of the change. In this case, we can retain the commit history in the PR while having a single clean commit to the base branch without noise with Squash and Merge option.

The caveat is that we do not want to abuse it. We want to make sure we retain a good commit history in the PR if it makes sense to have the change history. We can do fixup, squash and so on. As long as we stick to good git practice, the Squash and Merge option works really well.

(2020-11-16)
