---
sidebar_position: 8
---

# 8. Deploying Jekyll Site to Github Page with Github Actions

I created my personal portfolio site with Jekyll. The site is simply chucked in the Github page. Now, it is time to use Github actions to deploy the site to the Github page repo every time I commit to master.

Here is the code. I am not using the Jekyll actions available out there. Instead, installing Jekyll and build it with commands. It probably takes longer, but this works for me.

As for the credential to be used in the Github action, we need to create a personal access token (Settingds -> Developer Settings -> Personal access token). Then, add the token to the Jekyll site repo. Then, the action can access to the token as below:

```bash
with:
  ACCESS_TOKEN: ${{ secrets.GIT_PAGE_DEPLOY }}
  ...
```

That's it. Now every commit to master builds the site and push to Github.io repo.

```yml
name: Deploy to GitHub Page
on:
  push:
    branches:
      - master
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v2
        with:
          persist-credentials: false

      - name: Install
        run: |
          cd code
          npm i
          cd ..
      - name: Unit-Tests
        run: |
          cd code
          npm run test-single-run
          cd ..
      - name: Build-Assets
        run: |
          cd code
          npm run deploy:prod
          cd ..
          pwd
      - name: Set-Base-Url
        run: |
          cd deploy-scripts
          node add-base-url.js
          cd ..
          pwd
      - name: Build-Jekyll
        run: |
          sudo apt-get install ruby-full
          sudo gem install bundler
          sudo bundle install
          bundle exec jekyll build
      - name: Deploy_To_GithubPage
        uses: JamesIves/github-pages-deploy-action@3.5.9
        with:
          ACCESS_TOKEN: ${{ secrets.GIT_PAGE_DEPLOY }}
          BRANCH: master # The branch the action should deploy to.
          FOLDER: _site # The folder the action should deploy.
          REPOSITORY_NAME: mydatahack/mydatahack.github.io
          TARGET_FOLDER: portfolio
          CLEAN: true # Automatically remove deleted files from the deploy branch
```
