#!/usr/bin/env bash
# Default is validation only. An administrator explicitly selects --apply.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
config_dir=/etc/ditserve-docker
config_file=$config_dir/daemon.json
unit_file=/etc/systemd/system/ditserve-docker.service
containerd_unit=/etc/systemd/system/ditserve-containerd.service
data_root=/mnt/ssd2/zyd/docker-data
marker=$data_root/.ditserve-docker-owner
marker_text='ditserve-docker v1: /etc/ditserve-docker/daemon.json'
socket=unix:///run/ditserve-docker/docker.sock
mode=${1:---check}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
[[ $# -le 1 && ( "$mode" == --check || "$mode" == --apply ) ]] || die 'Usage: install.sh [--check|--apply]'
for command in docker dockerd containerd ctr nvidia-container-runtime systemd-analyze python3; do
    command -v "$command" >/dev/null || die "Missing command: $command"
done
dockerd --validate --config-file "$source_dir/daemon.json"
containerd --config "$source_dir/containerd.toml" config dump >/dev/null
systemd-analyze verify "$source_dir/ditserve-docker.service" "$source_dir/ditserve-containerd.service"
if [[ "$mode" == --check ]]; then
    printf '%s\n' 'Configuration validation passed. No system files, groups or services changed.' \
        'Apply after review with: sudo ./docker/4090-daemon/install.sh --apply'
    exit 0
fi
[[ $EUID -eq 0 ]] || die '--apply requires an administrator running as root.'
id zyd >/dev/null || die 'The local user zyd does not exist.'
mountpoint --quiet /mnt/ssd2 || die '/mnt/ssd2 is not mounted; refusing to write to the system disk.'
[[ $(findmnt -n -o TARGET --target /mnt/ssd2) == /mnt/ssd2 ]] || die 'Unexpected data disk mount.'
[[ $(findmnt -n -o FSTYPE --target /mnt/ssd2) == ext4 ]] || die 'This reviewed setup expects the existing ext4 data disk.'

# Refuse conflicting files, links or unknown existing data before any mutation.
python3 - "$source_dir" <<'PY'
import os
import grp
from pathlib import Path
import stat
import sys

source = Path(sys.argv[1])
config = Path('/etc/ditserve-docker/daemon.json')
units = [Path('/etc/systemd/system') / name for name in ('ditserve-docker.service', 'ditserve-containerd.service')]
data = Path('/mnt/ssd2/zyd/docker-data')
marker = data / '.ditserve-docker-owner'
expected_marker = b'ditserve-docker v1: /etc/ditserve-docker/daemon.json\n'

def require(ok, detail):
    if not ok:
        raise SystemExit('Refusing to install: ' + detail)

for ancestor in data.parents:
    if not ancestor.exists():
        continue
    st = ancestor.stat()
    require(not (st.st_mode & stat.S_IWOTH and not st.st_mode & stat.S_ISVTX),
            f'{ancestor} is world-writable without the sticky bit; an administrator must protect it first')
for path in (config.parent, data.parent, data, data / 'tmp', data / 'containerd', data / 'containerd/tmp'):
    require(str(path.resolve()) == str(path), f'{path} contains a symlink')
    if path.exists():
        st = path.stat()
        require(stat.S_ISDIR(st.st_mode) and st.st_uid == 0, f'{path} must be a root-owned directory')
        require(st.st_mode & 0o022 == 0, f'{path} must not be group/other writable')
targets = [config, config.parent / 'containerd.toml', *units]
for target in targets:
    template = source / target.name
    require(not target.is_symlink(), f'{target} is a symlink')
    if target.exists():
        st = target.stat()
        require(stat.S_ISREG(st.st_mode) and st.st_uid == 0 and not st.st_mode & 0o022,
                f'{target} is not a protected root-owned regular file')
        require(target.read_bytes() == template.read_bytes(), f'{target} differs from the reviewed template')
for unit in units:
    for base in ('/run/systemd/system', '/usr/local/lib/systemd/system', '/usr/lib/systemd/system', '/lib/systemd/system'):
        require(not os.path.lexists(Path(base) / unit.name), f'a pre-existing {unit.name} exists under {base}')
    for base in ('/etc/systemd/system', '/run/systemd/system', '/usr/local/lib/systemd/system', '/usr/lib/systemd/system', '/lib/systemd/system'):
        require(not os.path.lexists(Path(base) / (unit.name + '.d')), f'pre-existing service overrides under {base}')
require(not marker.is_symlink(), f'{marker} is a symlink')
if data.exists() and any(data.iterdir()):
    require(marker.is_file() and marker.read_bytes() == expected_marker,
            f'{data} is nonempty and is not marked as this daemon\'s data')
if marker.exists():
    require(marker.stat().st_uid == 0 and marker.read_bytes() == expected_marker,
            'unexpected data ownership marker')
for runtime in (Path('/run/ditserve-docker'), Path('/run/ditserve-containerd')):
    require(not runtime.is_symlink(), f'{runtime} is a symlink')
    if runtime.exists():
        require(runtime.is_dir() and runtime.stat().st_uid == 0, f'{runtime} must be root-owned')
        require(all(path.exists() for path in targets), 'runtime directory exists without matching installed configuration')
try:
    group = grp.getgrnam('ditserve-docker')
except KeyError:
    pass
else:
    require(set(group.gr_mem) <= {'zyd'}, 'ditserve-docker group has unexpected members')
PY

for unit in ditserve-docker.service ditserve-containerd.service; do
    dropins=$(systemctl show --property=DropInPaths --value "$unit")
    [[ -z "$dropins" ]] || die "The dedicated $unit has active overrides; review them manually."
done

# Install only absent dedicated files. Matching installations are left intact.
if [[ ! -d "$config_dir" ]]; then install -d -o root -g root -m 0755 "$config_dir"; fi
if [[ ! -e "$config_file" ]]; then install -o root -g root -m 0644 "$source_dir/daemon.json" "$config_file"; fi
if [[ ! -e "$config_dir/containerd.toml" ]]; then install -o root -g root -m 0644 "$source_dir/containerd.toml" "$config_dir/containerd.toml"; fi
if [[ ! -e "$unit_file" ]]; then install -o root -g root -m 0644 "$source_dir/ditserve-docker.service" "$unit_file"; fi
if [[ ! -e "$containerd_unit" ]]; then install -o root -g root -m 0644 "$source_dir/ditserve-containerd.service" "$containerd_unit"; fi
if [[ ! -d /mnt/ssd2/zyd ]]; then install -d -o root -g root -m 0755 /mnt/ssd2/zyd; fi
if [[ ! -d "$data_root" ]]; then install -d -o root -g root -m 0700 "$data_root"; fi
if [[ ! -e "$marker" ]]; then
    (umask 077; printf '%s\n' "$marker_text" > "$marker")
fi
if [[ ! -d "$data_root/tmp" ]]; then install -d -o root -g root -m 0700 "$data_root/tmp"; fi
if [[ ! -d "$data_root/containerd" ]]; then install -d -o root -g root -m 0700 "$data_root/containerd"; fi
if [[ ! -d "$data_root/containerd/tmp" ]]; then install -d -o root -g root -m 0700 "$data_root/containerd/tmp"; fi
if ! getent group ditserve-docker >/dev/null; then groupadd --system ditserve-docker; fi
case " $(id -nG zyd) " in
    *' ditserve-docker '*) ;;
    *) usermod -aG ditserve-docker zyd ;;
esac
systemctl daemon-reload
systemctl start ditserve-docker.service
for unit in ditserve-containerd.service ditserve-docker.service; do
    systemctl is-active --quiet "$unit" || die "$unit did not become active."
done

# Verify the real Docker store and private containerd before importing images.
actual=$(env -u DOCKER_CONTEXT -u DOCKER_TLS_VERIFY -u DOCKER_CERT_PATH \
    docker --host "$socket" info --format '{{.DockerRootDir}} {{.Driver}}')
[[ "$actual" == "$data_root overlay2" ]] || die "Unexpected daemon storage: $actual"
ctr --address /run/ditserve-containerd/containerd.sock version
printf '%s\n' 'Dedicated daemon started; no shared Docker/containerd service was changed.' \
    'Docker uses overlay2 on the data disk and the explicitly configured private containerd.' \
    'The service is not enabled at boot. zyd must reconnect to acquire the new group.' \
    'Use docker --host unix:///run/ditserve-docker/docker.sock for this daemon.'
