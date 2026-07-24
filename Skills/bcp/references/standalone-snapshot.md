# Direct timestamped project copy

Copy the exact named project directly to its final dated folder. Do not create
a `.partial` directory and do not use rsync.

```bash
root=/absolute/project/root
project_name=$(basename "$root")
destination=/data/backup/${project_name}_$(date +%m%d%y)

mountpoint -q /data/backup
test "$(findmnt -n -o FSTYPE -T /data/backup)" = ext4
mkdir -p "$destination"
cp -a --preserve=all "$root/." "$destination/"
```

When the user specifies another mounted backup-disk path, use that exact path.
An existing destination is merged and matching paths are overwritten; `cp`
does not remove stale destination-only files.
