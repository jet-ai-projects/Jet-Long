# Contributing

We welcome community contributions, which we accept via pull requests. To ensure
that all contributions to this project are properly licensed, we require that
every contributor sign off on their commits to indicate agreement with the
Developer Certificate of Origin (DCO).

## Submitting a pull request

1. Fork the repository and clone your fork locally.
2. Create a branch for your change:

   ```bash
   git checkout -b my-change
   ```

3. Make your changes and commit them with a DCO sign-off (see below):

   ```bash
   git commit -s -m "Describe your change"
   ```

4. Push the branch to your fork:

   ```bash
   git push origin my-change
   ```

5. Open a pull request against the `main` branch of this repository.

Every commit in your pull request must carry a `Signed-off-by` line, or the
automated DCO check will block the merge.

## Developer Certificate of Origin (DCO)

The DCO is a lightweight way for contributors to certify that they wrote or
otherwise have the right to submit the code they are contributing. You can read
the full text of the DCO at <https://developercertificate.org/>.

By contributing to this repository, you certify the terms of the DCO for every
commit you submit.

### How to sign off on your commits

To signify your agreement with the DCO, add a `Signed-off-by` line to **every**
Git commit message. The line must use your real name and an email address you
can be reached at — anonymous or pseudonymous sign-offs are not accepted.

First, make sure your Git identity is configured:

```bash
git config --global user.name "Jane Doe"
git config --global user.email "jane.doe@example.com"
```

Then pass the `-s` (or `--signoff`) flag when you commit:

```bash
git commit -s -m "Add a new feature"
```

Git will append a line to your commit message that looks like this:

```
Add a new feature

Signed-off-by: Jane Doe <jane.doe@example.com>
```

The name and email in the `Signed-off-by` line **must match** the commit
author. If they do not match, the automated DCO check will fail and your pull
request cannot be merged.

### Fixing commits that are missing a sign-off

If you forgot to sign off on your most recent commit:

```bash
git commit --amend -s --no-edit
```

If several commits in your branch are missing sign-offs, rebase over them and
sign each one. For example, to fix the last three commits:

```bash
git rebase --signoff HEAD~3
```

After amending or rebasing, you will need to force-push your branch:

```bash
git push --force-with-lease
```
