---
slug: data-engineering/data-ingestion/download-github-gist
title: Downloading All Public GitHub Gist Files
tags:
  [Data Engineering, Data Ingestion, API Data Ingestion, GitHub Gist, PythonL]
---

I used to use plug-ins to render code blocks for this blog. Yesterday, I decided to move all the code into GitHub Gist and inject them from there. <!-- truncate --> Using a WordPress plugin to render code blocks can be problematic when update happens. Plugins might not be up to date. It can break the site as most of the plugins are server-site rendering. That is exactly what happened to me with WordPress 5.2 update.

Now I am in the process of moving all the code examples to GitHub Gist. In this way, I don’t need to worry about plugins and give greater accessibility to those snippets.

To download all the files from Git Gist, you need to write a small code. This cannot be done from UI. Luckily, GitHub Gist version 3 API is pretty nice. This task can be done easily.

Here is the example of Python code that downloads all my GitHub Gist files. I’m happy for you to use or modify any code snippets for your own projects. You can also use it for your own gists by changing the username.

```python
# Download all public gist for a user
# by using v3 gist api (https://developer.github.com/v3/gists/)

import requests, json
headers = {"content-type" : "application/json"}
url = 'https://api.github.com/users/mydatahack/gists'
r = requests.get(url, headers = headers)
metadata_file = './data/my_gist_list.json'
# Getting metadata
prettyJson = json.dumps(r.json(), indent=4, sort_keys=True)
f = open(metadata_file, 'w')
f.write(prettyJson)

print('Metadata obtained as {}'.format(metadata_file))

# Downloading files
data = r.json()
counter = 0
for i in data:
    files_node = i['files']
    file_name = [k for k in files_node][0]
    r = requests.get(files_node[file_name]['raw_url'])
    f = open('./data/{}'.format(file_name), 'w')
    f.write(r.text)
    f.close()
    print('Downloaded {}'.format(file_name))
    counter += 1

print('{} files successfully downloaded.'.format(counter))
```
