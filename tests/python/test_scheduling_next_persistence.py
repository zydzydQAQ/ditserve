"""CPU-only behavioral checks for bounded asynchronous experiment artifacts."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from ditserve.scheduling.next_persistence import AsyncArtifacts


class AsyncArtifactsTests(unittest.TestCase):
    def test_constructor_leaves_replay_directory_creation_to_runner(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "replay"
            writer = AsyncArtifacts(path)
            try:
                self.assertFalse(path.exists())
                path.mkdir()
                paths = writer.save("r001", np.arange(4, dtype=np.float32), None)
                stats = writer.drain()
                self.assertEqual(set(paths), {"output"})
                self.assertTrue(Path(paths["output"]).is_absolute())
                np.testing.assert_array_equal(np.load(paths["output"], allow_pickle=False), np.arange(4))
                self.assertEqual((stats["submitted"], stats["completed"], stats["files"], stats["pending"]),
                                 (1, 1, 1, 0))
            finally:
                writer.close()

    def test_noncontiguous_readonly_arrays_roundtrip_without_input_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            output = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
            latent = np.arange(12, dtype=np.float16).reshape(3, 4)
            output.setflags(write=False)
            latent.setflags(write=False)
            originals = (output.copy(), latent.copy())
            writer = AsyncArtifacts(root)
            try:
                paths = writer.save("r000", output, latent)
                stats = writer.drain()
                for kind, array, before in zip(("output", "latent"), (output, latent), originals):
                    np.testing.assert_array_equal(np.load(paths[kind], allow_pickle=False), before)
                    np.testing.assert_array_equal(array, before)
                    self.assertFalse(array.flags.writeable)
                self.assertEqual(stats["files"], 2)
                self.assertEqual(stats["bytes"], sum(Path(path).stat().st_size for path in paths.values()))
                self.assertGreaterEqual(stats["write_ms"], 0.)
            finally:
                writer.close()

    def test_capacity_counts_running_plus_queued_and_records_backpressure(self):
        with tempfile.TemporaryDirectory() as root:
            entered, release, third_started, third_done = (threading.Event() for _ in range(4))
            real_save = np.save
            worker_threads, errors = set(), []

            def blocked_save(path, array, **kwargs):
                worker_threads.add(threading.get_ident())
                if "r000-output" in path:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release first write")
                return real_save(path, array, **kwargs)

            writer = AsyncArtifacts(root)

            def third_save():
                third_started.set()
                try:
                    writer.save("r002", np.arange(2), None)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    third_done.set()

            third = threading.Thread(target=third_save)
            try:
                with mock.patch("ditserve.scheduling.next_persistence.np.save", side_effect=blocked_save):
                    writer.save("r000", np.arange(2), None)
                    self.assertTrue(entered.wait(2))
                    writer.save("r001", np.arange(2), None)
                    third.start()
                    self.assertTrue(third_started.wait(2))
                    self.assertFalse(third_done.wait(.05))
                    release.set()
                    third.join(2)
                    self.assertFalse(third.is_alive())
                    stats = writer.drain()
                self.assertEqual(errors, [])
                self.assertEqual(stats["completed"], 3)
                self.assertEqual(stats["max_in_flight"], 2)
                self.assertGreater(stats["queue_wait_ms"], 0.)
                self.assertEqual(len(worker_threads), 1)
                self.assertNotIn(threading.get_ident(), worker_threads)
            finally:
                release.set()
                if third.ident is not None:
                    third.join(2)
                writer.close()

    def test_async_failure_reappears_on_save_drain_close_and_thread_stops(self):
        with tempfile.TemporaryDirectory() as root:
            entered, release = threading.Event(), threading.Event()
            writer = AsyncArtifacts(root)

            def failing_save(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release failed write")
                raise OSError("simulated disk failure")

            try:
                with mock.patch("ditserve.scheduling.next_persistence.np.save", side_effect=failing_save) as save:
                    writer.save("r000", np.arange(2), None)
                    self.assertTrue(entered.wait(2))
                    writer.save("r001", np.arange(2), None)
                    release.set()
                    with self.assertRaisesRegex(RuntimeError, "artifact persistence failed") as caught:
                        writer.drain()
                    self.assertIsInstance(caught.exception.__cause__, OSError)
                    self.assertEqual(str(caught.exception.__cause__), "simulated disk failure")
                    with self.assertRaisesRegex(RuntimeError, "artifact persistence failed"):
                        writer.save("r002", np.arange(2), None)
                    with self.assertRaisesRegex(RuntimeError, "artifact persistence failed"):
                        writer.close()
                    self.assertFalse(writer._thread.is_alive())
                    # The accepted second batch is cancelled after the first error.
                    self.assertEqual(save.call_count, 1)
                    with self.assertRaises(RuntimeError):
                        writer.close()
            finally:
                release.set()
                try:
                    writer.close()
                except RuntimeError:
                    pass

    def test_close_rejects_waiting_submitter_and_waits_for_accepted_batches(self):
        with tempfile.TemporaryDirectory() as root:
            entered, release, third_started, third_done, closed = (threading.Event() for _ in range(5))
            real_save = np.save
            writer = AsyncArtifacts(root)
            errors = []

            def blocked_save(path, array, **kwargs):
                if "r000-output" in path:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release first write")
                return real_save(path, array, **kwargs)

            def submit_third():
                third_started.set()
                try:
                    writer.save("r002", np.arange(2), None)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    third_done.set()

            def close_writer():
                try:
                    writer.close()
                finally:
                    closed.set()

            third = threading.Thread(target=submit_third)
            closer = threading.Thread(target=close_writer)
            try:
                with mock.patch("ditserve.scheduling.next_persistence.np.save", side_effect=blocked_save):
                    writer.save("r000", np.arange(2), None)
                    self.assertTrue(entered.wait(2))
                    writer.save("r001", np.arange(2), None)
                    third.start()
                    self.assertTrue(third_started.wait(2))
                    self.assertFalse(third_done.wait(.02))
                    closer.start()
                    self.assertTrue(third_done.wait(2))
                    self.assertFalse(closed.wait(.02))
                    release.set()
                    closer.join(2)
                    third.join(2)
                self.assertFalse(closer.is_alive())
                self.assertFalse(writer._thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertRegex(str(errors[0]), "writer is closed")
                stats = writer.drain()
                self.assertEqual((stats["submitted"], stats["completed"], stats["closed"]), (2, 2, True))
                self.assertGreater(stats["drain_wait_ms"], 0.)
                with self.assertRaisesRegex(RuntimeError, "writer is closed"):
                    writer.save("r003", np.arange(2), None)
            finally:
                release.set()
                if third.ident is not None:
                    third.join(2)
                if closer.ident is not None:
                    closer.join(2)
                writer.close()

    def test_rejects_non_numpy_object_arrays_path_escape_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as root:
            writer = AsyncArtifacts(root)
            try:
                with self.assertRaises(TypeError):
                    writer.save("list", [1, 2], None)
                with self.assertRaises(TypeError):
                    writer.save("object", np.array([object()], dtype=object), None)
                for rid in ("", "..", "../outside", "dir/file"):
                    with self.subTest(rid=rid), self.assertRaises(ValueError):
                        writer.save(rid, np.arange(2), None)
                writer.save("r000", np.arange(2), None)
                with self.assertRaisesRegex(ValueError, "already"):
                    writer.save("r000", np.arange(2), None)
                writer.drain()
                self.assertEqual(sorted(p.name for p in Path(root).iterdir()), ["r000-output.npy"])
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
