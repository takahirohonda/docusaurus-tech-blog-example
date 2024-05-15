---
sidebar_position: 7
---

# 7. How to Add Private Key to SSH Agent with Git Bash

The ssh-agent provide a secure way to hold the private keys of remote server. It is a program that runs in the background. While you are logged into the system, it is storing your keys in memory. If you are using a SSH key with Git, the ssh-agent is used to authenticate from the local machine and access repositories. This is a quick cheat sheet for adding your private key to the ssh-agent.

```bash
# First, you need to start the ssh-agent in background
eval $(ssh-agent)

# Then use ssh-add to add the key
ssh-add /c/users/mydatahack/.ssh/mydatahack_id_rsa
```

You should be able to authenticate with the correct private key now.

(2020-03-20)
