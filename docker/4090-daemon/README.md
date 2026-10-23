# 本机专用 Docker 服务

管理员已于 2026-10-19 在当前双 4090 主机应用这组配置，两个专用服务均已启动。实际 DockerRootDir 为 `/mnt/ssd2/zyd/docker-data`，存储驱动为 `overlay2`，迁移镜像已导入并核对 ID。已按本机 Docker 29.1.3、containerd 2.2.1、systemd 255 和 ext4 数据盘检查。已有 `docker.service`、`containerd.service` 和各自的数据保持独立。

镜像层和容器运行时持久存储使用 **`/mnt/ssd2/zyd/docker-data`**。SSD2 挂载根为 0755，管理员会创建 root 所有的专用目录。模型、镜像归档、项目结果和模型缓存仍使用 `/mnt/ssd0/zyd/` 下的现有目录；不修改任何共享盘挂载根权限。

| 用途 | 本方案路径 |
| --- | --- |
| Docker 客户端 socket | `/run/ditserve-docker/docker.sock` |
| Docker 持久数据、镜像层、容器日志 | `/mnt/ssd2/zyd/docker-data` |
| Docker 运行状态、PID | `/run/ditserve-docker/exec`、`/run/ditserve-docker/dockerd.pid` |
| containerd 持久数据 | `/mnt/ssd2/zyd/docker-data/containerd` |
| containerd 运行状态、socket | `/run/ditserve-containerd`、`/run/ditserve-containerd/containerd.sock` |
| 独立配置 | `/etc/ditserve-docker/daemon.json`、`/etc/ditserve-docker/containerd.toml` |
| 独立服务 | `ditserve-docker.service`、`ditserve-containerd.service` |

Docker 29 的新安装默认使用 containerd image store，仅改 `data-root` 不能保证 containerd 内容一起迁移。本方案明确使用 `containerd-snapshotter=false`、经典 `overlay2`，并显式连接独立 containerd；后者的 `root`、`state`、临时目录和 opt 插件目录均独立配置，禁用 CRI/NRI，且不导入系统 containerd 配置。[Docker 数据目录说明](https://docs.docker.com/engine/daemon/#daemon-data-directory)、[containerd image store 说明](https://docs.docker.com/engine/storage/containerd/)

Docker 29.1.3 在没有显式 containerd 地址时会自动探测系统 socket；本方案使用明确的私有 socket，避免依赖自动探测。两个服务均由 systemd 管理；本机驱动检查仍保持项目的 `MIN_DRIVER=560`。[Moby 29.1.3 实现](https://github.com/moby/moby/blob/docker-v29.1.3/daemon/command/daemon.go#L960)

管理员审阅这两个配置、两个 service 和 `install.sh` 后，在仓库根目录执行：

```bash
./docker/4090-daemon/install.sh --check
sudo ./docker/4090-daemon/install.sh --apply
```

安装器不会修改共享数据盘根权限；如果祖先目录是无 sticky 的全员可写目录，就会拒绝安装。它仅创建专用 `ditserve-docker` 组、将 `zyd` 加入组、安装专用配置/服务、创建 root 所有的 `/mnt/ssd2/zyd`（0755）和 Docker 数据目录（0700），并启动专用服务。配置或 unit 已存在时必须逐字一致；未知非空数据目录、符号链接及 service overrides 会被拒绝。重复执行不会重启已经运行的服务，也不会清空数据。若启动或后置校验失败，保留专用服务状态供检查，可用下面的诊断/停止命令处理。

`zyd` 重新登录后获得组权限。该组通过 rootful Docker socket 获得宿主 root 等价能力，管理员应按这一权限范围授权。Docker 官方仍将同机多 daemon 标为实验性；本方案关闭 bridge、iptables/ip6tables、IP forwarding、IP masquerading 和 userland proxy，以减少与共享服务的网络冲突。[多 daemon 官方说明](https://docs.docker.com/reference/cli/dockerd/#run-multiple-daemons)

安装器仅启动服务，不设置开机自启。需要自启时，由管理员另行运行 `sudo systemctl enable ditserve-docker.service`；其依赖会启动专用 containerd。服务启动需 `/mnt/ssd2` 已挂载。少量 systemd 服务日志及 `/etc` 配置属于宿主系统；镜像、容器元数据、容器日志、containerd 持久数据和服务临时文件位于 SSD2 数据盘。

管理员和重新登录后的 `zyd` 使用显式 socket；容器需显式指定 `--network none` 或 `--network host`，本项目双卡 `torchrun` 使用 host 网络：

```bash
docker --host unix:///run/ditserve-docker/docker.sock info \
  --format 'Root={{.DockerRootDir}} Driver={{.Driver}}'
# 预期 Root=/mnt/ssd2/zyd/docker-data Driver=overlay2

docker --host unix:///run/ditserve-docker/docker.sock load \
  -i /mnt/ssd0/zyd/images/ditserve-rewrite-runtime.tar.gz

docker --host unix:///run/ditserve-docker/docker.sock run --rm --network none \
  ditserve:rewrite-runtime ditserve --version
```

归档来源是 `zyd@172.31.225.239:/home/zyd/code/dit-rewrite/delivery/ditserve-rewrite-runtime.tar.gz`，大小 `3757147317` 字节，SHA256 为 `e98bb0e65f9d32923e457faee05c79e05a3ba72b851548cd081b0bae7580ff2a`。本方案安装器不会导入镜像或运行 GPU。

诊断、停止命令只指向专用服务：

```bash
sudo systemctl status ditserve-docker.service ditserve-containerd.service
sudo journalctl -u ditserve-docker.service -u ditserve-containerd.service -n 100 --no-pager
sudo systemctl stop ditserve-docker.service ditserve-containerd.service
```

准备阶段的验证包括 `bash -n`、`dockerd --validate`、`containerd config dump`、`systemd-analyze verify`，以及启动包装脚本的假 Docker 参数测试。应用后已确认服务启动、数据目录、镜像 ID、容器内依赖和双卡初始化；BF16、Flash SDPA、Triton 和 NCCL 小张量检查均通过。模型验收状态见 [本机部署记录](../../docs/rtx4090-local.md)。
