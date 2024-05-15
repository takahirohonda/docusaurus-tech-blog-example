---
sidebar_position: 10
---

# 10. How to specify which Node version to use in Github Actions

When you want to specify which Node version to use in your Github Actions, you can use actions/setup-node@v2.

```yml
name: Test and Build from feature branch
on:
  push:
    branches-ignore:
      - "main"
jobs:
  test-and-build-feature-branch:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu-latest]
        node-version: [14.x]
    steps:
      - name: Checkout
        uses: actions/checkout@v2
        with:
          persist-credentials: false
      - name: Use Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v2
        with:
          node-version: ${{ matrix.node-version }}
      - name: Install
        run: |
          yarn install
      - name: Type Check
        run: |
          yarn type-check
      - name: Lint
        run: |
          yarn lint
      - name: Unit Tests
        run: |
          yarn test
      - name: Module Build
        run: |
          yarn test
      - name: Build
        run: |
          yarn build
```

The alternative way is to use a node container.

When you try to use a publicly available node container like runs-on: node:alpine-xx, the pipeline gets stuck in a queue. runs-on is not for a container name. It’s for a virtual machine hosted by GitHub and the best option for a Linux machine at the moment is ubuntu-latest.

In fact, GitHub supports runners for Ubuntu Linux (ubuntu-latest), Mac (mac-latest) and Windows (windows-latest). You can see other supported virtual environments here.

Instead of adding a container name in runs-on, we can add container: [image-name]. Then the jobs will be run on the specified container.

```yml
name: Run Jobs in a container
on:
  push:
    branches:
      - main
jobs:
  run-in-container:
    runs-on: ubuntu-latest
    container: node:14-alpine
    steps:
      - name: Checkout
        uses: actions/checkout@v2
        with:
          persist-credentials: false
      - name: Install
        run: |
          yarn install
      - name: Unit Tests
        run: |
          yarn test
      - name: Build
        run: |
          yarn build
```

(2022-03-04)
