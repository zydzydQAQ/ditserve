"""Bind xFuser to local or prebuilt request-scoped process groups."""

import datetime

import torch

from xfuser.core.distributed import parallel_state as ps
from xfuser.core.distributed.group_coordinator import (
    GroupCoordinator,
    PipelineGroupCoordinator,
    SequenceParallelGroupCoordinator,
)

from .process_group import PROCESS_GROUP


class ProcessGroupPool:
    def __init__(self, rank, world_size, device, backend, groups=None):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.backend = backend
        self.groups = groups
        self.metadata = {}
        self.current_ranks = None

    @classmethod
    def local(cls, device="cpu"):
        return cls(0, 1, device, "gloo")

    @classmethod
    def distributed(cls, groups, device):
        if not torch.distributed.is_initialized():
            raise ValueError("xDiT distributed execution requires an initialized process group")
        pool = cls(groups.rank, torch.distributed.get_world_size(), device, groups.backend, groups)
        for ranks in groups.groups:
            pool.metadata[ranks] = torch.distributed.new_group(
                list(ranks), backend="gloo",
                timeout=datetime.timedelta(seconds=groups.timeout_seconds))
        return pool

    @property
    def distributed_world(self):
        return self.groups is not None

    def _device_group(self, ranks):
        if self.groups is None:
            return None
        return self.groups.group(tuple(ranks))

    def _cpu_group(self, ranks):
        if self.groups is None:
            return None
        return self.metadata[tuple(ranks)]

    def _coordinator(self, ranks, *, sequence=False, pipeline=False, ulysses_degree=None, ring_degree=1):
        ranks = tuple(ranks)
        if self.rank not in ranks:
            raise ValueError("local rank is not in the requested xDiT group")
        cls = (SequenceParallelGroupCoordinator if sequence else
               PipelineGroupCoordinator if pipeline else GroupCoordinator)
        coordinator = cls.__new__(cls)
        coordinator.rank = self.rank
        coordinator.local_rank = self.rank
        coordinator.ranks = ranks
        coordinator.world_size = len(ranks)
        coordinator.rank_in_group = ranks.index(self.rank)
        coordinator.device_group = self._device_group(ranks)
        coordinator.cpu_group = self._cpu_group(ranks)
        coordinator.shm_broadcaster = None
        coordinator.device = self.device
        if sequence:
            degree = len(ranks)
            r_deg = int(ring_degree or 1)
            u_deg = int(ulysses_degree or (degree // r_deg))
            if u_deg * r_deg != degree:
                raise ValueError(f"Invalid 2D USP degrees: ulysses={u_deg}, ring={r_deg}, world={degree}")
            if r_deg == 1:
                coordinator.ulysses_group = coordinator.device_group
                coordinator.ring_group = self._device_group((self.rank,))
                coordinator.ulysses_world_size = degree
                coordinator.ulysses_rank = coordinator.rank_in_group
                coordinator.ring_world_size = 1
                coordinator.ring_rank = 0
            else:
                # 2D Hybrid USP for dual-socket PCIe: intra-socket Ulysses (contiguous u_deg ranks)
                # + inter-socket Ring (strided by u_deg across sockets).
                u_idx = coordinator.rank_in_group // u_deg
                u_sub = tuple(ranks[u_idx * u_deg:(u_idx + 1) * u_deg])
                r_off = coordinator.rank_in_group % u_deg
                r_sub = tuple(ranks[i * u_deg + r_off] for i in range(r_deg))
                coordinator.ulysses_group = self._device_group(u_sub) if self.groups and u_sub in getattr(self.groups, "groups", ()) else coordinator.device_group
                coordinator.ring_group = self._device_group(r_sub) if self.groups and r_sub in getattr(self.groups, "groups", ()) else coordinator.device_group
                coordinator.ulysses_world_size = u_deg
                coordinator.ulysses_rank = coordinator.rank_in_group % u_deg
                coordinator.ring_world_size = r_deg
                coordinator.ring_rank = u_idx
        if pipeline:
            coordinator.cpu_groups = []
            coordinator.device_groups = []
            coordinator.recv_buffer_set = False
            coordinator.dtype = None
            coordinator.num_pipefusion_patches = 1
            coordinator.skip_tensor_recv_buffer_set = False
            coordinator.skip_device_group = coordinator.device_group
            coordinator.reset_buffer()
        return coordinator

    def install(self):
        world = tuple(range(self.world_size))
        ps._WORLD = self._coordinator(world)
        ps._DP = self._coordinator((self.rank,))
        ps._CFG = self._coordinator((self.rank,))
        ps._TP = self._coordinator((self.rank,))
        ps._PP = self._coordinator((self.rank,), pipeline=True)
        ps._FS = self._coordinator((self.rank,))
        ps._VAE = None
        self.activate((self.rank,))

    def activate(self, sp_ranks, runtime_state=None, *, ulysses_degree=None, ring_degree=1):
        sp_ranks = tuple(sp_ranks)
        if not sp_ranks or self.rank not in sp_ranks:
            raise ValueError("xDiT SP group must contain the local rank")
        degree = len(sp_ranks)
        r_deg = int(ring_degree or 1)
        u_deg = int(ulysses_degree or (degree // r_deg))
        coordinator = self._coordinator(
            sp_ranks, sequence=True, ulysses_degree=u_deg, ring_degree=r_deg
        )
        ps._SP = coordinator
        ps._MODEL_REPLICA = self._coordinator(sp_ranks)
        ps._DIT = self._device_group(sp_ranks)
        PROCESS_GROUP.ULYSSES_PG = coordinator.ulysses_group
        PROCESS_GROUP.RING_PG = coordinator.ring_group
        self.current_ranks = sp_ranks
        if runtime_state is not None:
            config = runtime_state.parallel_config
            config.dp_config.dp_degree = 1
            config.dp_config.cfg_degree = 1
            config.dp_config.dit_parallel_size = degree
            config.sp_config.ulysses_degree = u_deg
            config.sp_config.ring_degree = r_deg
            config.sp_config.sp_degree = degree
            config.sp_config.dit_parallel_size = degree
            config.dp_degree = 1
            config.cfg_degree = 1
            config.dit_parallel_size = degree
            config.sp_degree = degree
            config.ulysses_degree = u_deg
            config.ring_degree = r_deg
            runtime_state.ready = False
        return coordinator

    def close(self):
        PROCESS_GROUP.ULYSSES_PG = None
        PROCESS_GROUP.RING_PG = None
        self.current_ranks = None
        for name in ("_WORLD", "_MODEL_REPLICA", "_DP", "_CFG", "_SP", "_PP", "_TP", "_FS", "_DIT", "_VAE"):
            setattr(ps, name, None)
        if self.groups is not None:
            for group in self.metadata.values():
                if group is not torch.distributed.GroupMember.NON_GROUP_MEMBER:
                    torch.distributed.destroy_process_group(group)
            self.metadata.clear()
