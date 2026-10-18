import os
import socket
import tempfile
import unittest

try:
    import torch
    import torch.multiprocessing as mp
except ImportError:  # pragma: no cover - plain python3 runs skip numeric tests
    torch = None


def _worker(rank, world, port, path):
    import json
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from ditserve.evaluation.collectives import measure
        rows = measure(sizes_mb=(0.25,), iterations=2, warmup=1, backend="gloo")
        if rank == 0:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(rows, stream)
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipIf(torch is None, "torch unavailable")
class CollectivesTests(unittest.TestCase):
    def test_gloo_contract_reports_each_aligned_group_and_mode(self):
        import json
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rows.json")
            mp.spawn(_worker, args=(4, port, path), nprocs=4, join=True)
            with open(path, encoding="utf-8") as stream:
                rows = json.load(stream)
        keys = {(tuple(row["group"]), row["mode"]) for row in rows}
        self.assertEqual(keys, {((0, 1), "concurrent"), ((2, 3), "concurrent"), ((0, 1), "isolated"),
                                ((0, 1, 2, 3), "concurrent"), ((0, 1, 2, 3), "isolated")})
        for row in rows:
            self.assertGreater(row["per_rank_gbps"], 0)
            self.assertAlmostEqual(row["per_rank_send_mb"], 0.25 * (row["degree"] - 1) / row["degree"])


if __name__ == "__main__":
    unittest.main()
