---
name: bcp
description: Mirror /data/projects with the shared rsync backup script or create a standalone project snapshot with cp, then optionally commit, push, and tag. Use for /bcp, backup/commit/push requests, timestamped project copies under /data/backup, and releases that require the backup to happen first.
---

# Backup, Commit, Push

Follow the user's named targets and operation order literally. If the user asks
to back up one project and release another, back up the named project first;
never reinterpret the release repository as the backup source.

Do not run builds, tests, smoke tests, benchmarks, or GPU work unless the user
explicitly requests them.

## Backup

Run the mirror script unconditionally whenever this skill is invoked:

```bash
bash /data/projects/backup.sh
```

The shared script uses `rsync --delete` to mirror `/data/projects/` into
`/data/backup/projects/`, shows aggregate progress only, and performs no Git
operation. It excludes the unreadable source-side `/.Trash-0/` system path and
deletes any matching excluded path from the destination.

For a dated or timestamped standalone copy, use `cp`, never rsync. Derive the
project name from the exact canonical project root and use the destination requested by the user, or
`/data/backup/{project_name}_MMDDYY` when none is specified. Confirm the backup
disk is mounted. Copy directly into the final folder with no `.partial`,
staging folder, rsync, manifest, or checksum pass. Existing files with matching
paths are overwritten; unrelated older files in the destination are retained:

```bash
root=/absolute/project/root
project_name=$(basename "$root")
destination=/data/backup/${project_name}_$(date +%m%d%y)
mkdir -p "$destination"
cp -a --preserve=all "$root/." "$destination/"
```

See [references/standalone-snapshot.md](references/standalone-snapshot.md) for
the exact direct-copy form.

## Commit and push

When requested, use the current repository and the simplest commands:

```bash
git add .
git commit -m "<short context-derived summary>"
git push
```

Use a short message from the known task context. Do not invent a detailed
message. If there is nothing to commit, do not create an empty commit.

For a requested release tag, create and push it only after the requested
backup:

```bash
git tag -a <version> -m "<project> <version>"
git push origin refs/tags/<version>
```

## Report

Report only the final backup path, commit hash when a commit was created,
pushed branch, and tag target. Mention any command that failed. Do not add
unrequested testing or process narration.
