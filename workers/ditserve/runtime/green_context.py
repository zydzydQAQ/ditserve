"""Lazy Green Context stream quotas; importing this module only uses the CPU."""

from contextlib import contextmanager
import hashlib


_ROLES = ("video", "image")
_MASK_EVIDENCE = (
    "Split-resource fingerprints are same-version driver resource identity diagnostics, "
    "not Nsight mask proof."
)


def _integer(name, value):
    if type(value) is not int:
        raise ValueError(name + " must be an integer")
    return value


def _invoke(driver, name, *arguments):
    result = getattr(driver, name)(*arguments)
    if not isinstance(result, tuple) or not result:
        raise RuntimeError(name + ": malformed cuda-bindings result")
    error, *values = result
    if int(error) != int(driver.CUresult.CUDA_SUCCESS):
        raise RuntimeError(name + ": " + str(error))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def partition_indices(total_sm, granularity, role, video_sm, image_sm):
    """Return split-group indices for one role without importing CUDA or torch."""
    if role not in _ROLES:
        raise ValueError("role must be 'video' or 'image'")
    for name, value in (("total_sm", total_sm), ("granularity", granularity),
                        ("video_sm", video_sm), ("image_sm", image_sm)):
        _integer(name, value)
    if total_sm <= 0:
        raise ValueError("total_sm must be positive")
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    if video_sm < 0 or image_sm < 0:
        raise ValueError("video_sm and image_sm must not be negative")
    if total_sm % granularity:
        raise ValueError("total_sm must be divisible by granularity")
    if video_sm % granularity or image_sm % granularity:
        raise ValueError("video_sm and image_sm must be divisible by granularity")

    group_count = total_sm // granularity
    if (video_sm, image_sm) == (total_sm, total_sm):
        return list(range(group_count))
    if video_sm + image_sm != total_sm:
        raise ValueError("video_sm + image_sm must equal total_sm, or both must equal total_sm")

    video_groups = video_sm // granularity
    if role == "video":
        return list(range(video_groups))
    return list(range(video_groups, video_groups + image_sm // granularity))


class GreenPool:
    """Own the Green Contexts and external torch streams for one worker role."""

    def __init__(self, role, partitions, default):
        if role not in _ROLES:
            raise ValueError("role must be 'video' or 'image'")
        if not isinstance(partitions, dict) or not partitions:
            raise ValueError("partitions must be a non-empty dict")
        if not isinstance(default, str) or default not in partitions:
            raise ValueError("default must name a partition")

        copied = {}
        for label, quotas in partitions.items():
            if not isinstance(label, str) or not label:
                raise ValueError("partition labels must be non-empty strings")
            if not isinstance(quotas, (tuple, list)) or len(quotas) != 2:
                raise ValueError("partition " + repr(label) + " must contain (video_sm, image_sm)")
            video_sm = _integer("video_sm", quotas[0])
            image_sm = _integer("image_sm", quotas[1])
            if video_sm < 0 or image_sm < 0:
                raise ValueError("partition quotas must not be negative")
            copied[label] = (video_sm, image_sm)

        self.role = role
        self.default = default
        self._partitions = copied
        self._initialized = False
        self._closed = False
        self._torch = None
        self._driver = None
        self._device = None
        self._physical_sm = None
        self._granularity_sm = None
        self._groups = []
        self._split_group_sm = []
        self._split_group_fingerprints = []
        self._entries = {}

    def initialize(self):
        if self._closed:
            raise RuntimeError("GreenPool is closed")
        if self._initialized:
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("GreenPool requires PyTorch with CUDA support") from error
        try:
            from cuda.bindings import driver
        except ImportError as error:
            raise RuntimeError("GreenPool requires cuda-bindings") from error

        self._torch, self._driver = torch, driver
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("GreenPool requires CUDA; torch.cuda.is_available() is false")
            if not torch.cuda.is_initialized():
                raise RuntimeError("GreenPool.initialize() requires the caller to initialize CUDA first")

            device = torch.cuda.current_device()
            self._device = device
            torch.cuda.current_stream(device).synchronize()
            physical = int(torch.cuda.get_device_properties(device).multi_processor_count)
            if physical < 2 or physical % 2:
                raise RuntimeError("GreenPool requires a positive even physical SM count; got " + str(physical))

            resource = _invoke(driver, "cuDeviceGetDevResource", device,
                               driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
            if int(resource.sm.smCount) != physical:
                raise RuntimeError("driver/torch SM mismatch: %d vs %d" % (resource.sm.smCount, physical))
            groups, count, remainder = _invoke(driver, "cuDevSmResourceSplitByCount",
                                               physical // 2, resource, 0, 2)
            count = int(count)
            if count < 0 or count > len(groups):
                raise RuntimeError("cuDevSmResourceSplitByCount returned invalid count " + str(count))
            groups = groups[:count]
            group_sm = [int(item.sm.smCount) for item in groups]
            remainder_sm = 0 if remainder is None else int(remainder.sm.smCount)
            if not group_sm or any(size <= 0 for size in group_sm):
                raise RuntimeError("invalid SM split: count=%d, sizes=%r" % (count, group_sm))
            granularity = group_sm[0]
            if any(size != granularity for size in group_sm):
                raise RuntimeError("asymmetric SM split groups: " + repr(group_sm))
            if sum(group_sm) + remainder_sm != physical:
                raise RuntimeError("SM split does not account for the physical device: %r + %d != %d" %
                                   (group_sm, remainder_sm, physical))
            if remainder_sm:
                raise RuntimeError("SM split has unsupported remainder resources: " + str(remainder_sm))

            self._groups = groups
            self._split_group_sm = group_sm
            self._split_group_fingerprints = [
                hashlib.sha256(bytes(item._internal_padding) + bytes(item._oversize)).hexdigest()
                for item in groups
            ]
            self._physical_sm = physical
            self._granularity_sm = granularity

            for label, quotas in self._partitions.items():
                requested = quotas[_ROLES.index(self.role)]
                indices = partition_indices(physical, granularity, self.role, *quotas)
                selected = [groups[index] for index in indices]
                entry = dict(requested_sm=requested, observed_sm=None, group_indices=indices,
                             resources=selected, descriptor=None, context=None, stream=None, external=None)
                self._entries[label] = entry
                if not selected:
                    raise RuntimeError("partition " + repr(label) + " requests zero SM for role " + self.role +
                                       "; a Green Context stream requires at least one SM group")

                entry["descriptor"] = _invoke(driver, "cuDevResourceGenerateDesc", selected, len(selected))
                context = _invoke(driver, "cuGreenCtxCreate", entry["descriptor"], device,
                                  int(driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
                entry["context"] = context
                observed_resource = _invoke(driver, "cuGreenCtxGetDevResource", context,
                                            driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
                observed = int(observed_resource.sm.smCount)
                if observed != requested:
                    raise RuntimeError("Green quota mismatch for %r: requested %d SM, observed %d SM" %
                                       (label, requested, observed))
                entry["observed_sm"] = observed

                stream = _invoke(driver, "cuGreenCtxStreamCreate", context,
                                 int(driver.CUstream_flags.CU_STREAM_NON_BLOCKING), 0)
                entry["stream"] = stream
                bound = _invoke(driver, "cuStreamGetGreenCtx", stream)
                if int(bound) != int(context):
                    raise RuntimeError("Green stream is not bound to its context for partition " + repr(label))
                entry["external"] = torch.cuda.ExternalStream(int(stream), device=device)

            self._initialized = True
        except BaseException as error:
            try:
                self._release()
            except Exception as cleanup_error:
                raise RuntimeError("GreenPool initialization failed: %s; cleanup failed: %s" %
                                   (error, cleanup_error)) from error
            raise

    @contextmanager
    def use(self, label=None):
        label = self.default if label is None else label
        if label not in self._partitions:
            raise ValueError("unknown GreenPool partition " + repr(label))
        self.initialize()
        stream = self._entries[label]["external"]
        with self._torch.cuda.stream(stream):
            yield stream

    def report(self):
        labels = {}
        quota_index = _ROLES.index(self.role)
        for label, quotas in self._partitions.items():
            entry = self._entries.get(label)
            labels[label] = dict(
                requested_sm=quotas[quota_index],
                observed_sm=None if entry is None else entry["observed_sm"],
                group_indices=[] if entry is None else list(entry["group_indices"]),
            )
        return dict(
            enabled=True,
            initialized=self._initialized,
            role=self.role,
            default=self.default,
            granularity_sm=self._granularity_sm,
            physical_sm=self._physical_sm,
            split_group_sm=list(self._split_group_sm),
            split_group_fingerprints=list(self._split_group_fingerprints),
            labels=labels,
            mask_evidence=_MASK_EVIDENCE,
            same_process_partitions_disjoint_by_construction=True,
            cross_process_masks_verified=None,
        )

    def _release(self):
        self._initialized = False
        entries = list(self._entries.items())
        for label, entry in entries:
            entry["external"] = None

        errors = []
        if self._driver is not None:
            for label, entry in entries:
                stream, entry["stream"] = entry["stream"], None
                if stream is not None:
                    try:
                        _invoke(self._driver, "cuStreamDestroy", stream)
                    except Exception as error:
                        errors.append("%s stream: %s" % (label, error))
            for label, entry in entries:
                context, entry["context"] = entry["context"], None
                if context is not None:
                    try:
                        _invoke(self._driver, "cuGreenCtxDestroy", context)
                    except Exception as error:
                        errors.append("%s context: %s" % (label, error))

        self._entries.clear()
        self._groups = []
        if errors:
            raise RuntimeError("GreenPool cleanup failed: " + "; ".join(errors))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._release()
