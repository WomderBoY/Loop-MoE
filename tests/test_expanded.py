import gzip
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from loop_moe.expanded import check_gpu, merge_model, preflight, prepare, run, sha256, validate_shard, worker_plan
from loop_moe.trace import SCHEMA_VERSION, TRACE_SCOPE, read_trace, write_line


class ExpandedTests(unittest.TestCase):
    def make_run(self, root):
        from loop_moe.expanded import save_json
        config = {"models": ["test"], "checkpoints": {"test": {
            "repo": "test/model", "revision": "fixed", "architecture": "foil", "dtype": "bfloat16",
        }}, "prompts": "prompts.jsonl", "max_new_tokens": 512, "shard_size": 2}
        save_json(root / "config.json", config)
        with (root / "prompts.jsonl").open("w") as stream:
            for i in range(3):
                write_line(stream, {"id": str(i), "text": "A prompt", "category": "text"})
        config, prompts_path, prompts, shards = prepare(root / "config.json", root / "out")
        for shard in shards:
            path = root / "out/shards" / shard.stem / "test/trace.jsonl.gz"
            path.parent.mkdir(parents=True)
            selected = prompts[int(shard.stem) * 2:(int(shard.stem) + 1) * 2]
            with gzip.open(path, "wt") as stream:
                write_line(stream, dict(config["checkpoints"]["test"], model="test",
                                        schema_version=SCHEMA_VERSION, trace_scope=TRACE_SCOPE,
                                        request_count=len(selected), prompts_sha256=sha256(shard),
                                        max_new_tokens=512))
                for prompt in selected:
                    write_line(stream, {"request_id": prompt["id"]})
        return config, prompts_path, shards

    def test_merge_preserves_order_and_full_workload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prompts, shards = self.make_run(root)
            path = merge_model(config, "test", shards, prompts, root / "out")
            meta, requests = read_trace(path)
            self.assertEqual(meta["request_count"], 3)
            self.assertEqual(meta["prompts_sha256"], sha256(prompts))
            self.assertEqual([r["request_id"] for r in requests], ["0", "1", "2"])
            self.assertEqual(merge_model(config, "test", shards, prompts, root / "out"), path)

    def test_resume_rejects_changed_inputs_and_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prompts, shards = self.make_run(root)
            shard = shards[0]
            path = root / "out/shards/000/test/trace.jsonl.gz"
            shard.write_text(shard.read_text().replace('"id":"0"', '"id":"different"') + "\n")
            with self.assertRaisesRegex(ValueError, "Wrong workload"):
                validate_shard(path, config, "test", shard)
            prompts.write_text(prompts.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Run inputs/code changed"):
                prepare(root / "config.json", root / "out")

    def test_missing_shard_never_publishes_partial_merged_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prompts, shards = self.make_run(root)
            (root / "out/shards/001/test/trace.jsonl.gz").unlink()
            with self.assertRaises(FileNotFoundError):
                merge_model(config, "test", shards, prompts, root / "out")
            self.assertFalse((root / "out/test/trace.jsonl.gz").exists())

    @patch("loop_moe.expanded.subprocess.run")
    def test_busy_gpu_is_rejected_even_at_zero_utilization(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "GPU-test, 123\n", "")
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            check_gpu({"index": "5", "uuid": "GPU-test"})
        check_gpu({"index": "5", "uuid": "GPU-test"}, own_pid=123)
        run.return_value = subprocess.CompletedProcess([], 0, "GPU-test, 123\nGPU-test, 456\n", "")
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            check_gpu({"index": "5", "uuid": "GPU-test"}, own_pid=123)

    def test_chat_preflight_counts_input_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"max_position_embeddings": 600}')
            config = {"models": ["loopie"], "max_new_tokens": 512,
                      "checkpoints": {"loopie": {"architecture": "loopie"}}}
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: SimpleNamespace(
                input_ids=SimpleNamespace(shape=(1, 176))))
            factory = SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
            with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=factory)}), \
                    patch("loop_moe.collect.download_model", return_value=root):
                with self.assertRaisesRegex(ValueError, "exceeds context"):
                    preflight(config, [{"id": "long-chat", "text": "Prompt text"}])
                (root / "config.json").write_text('{"max_position_embeddings": 1024}')
                result = preflight(config, [{"id": "long-chat", "text": "Prompt text"}])
                self.assertEqual(result[0]["max_prompt_tokens"], 176)

    @patch("loop_moe.expanded.subprocess.run")
    def test_parallel_models_allow_only_the_drivers_own_pids(self, run):
        gpu = {"index": "0", "uuid": "GPU-test"}
        run.return_value = subprocess.CompletedProcess([], 0,
            "GPU-test, 101\nGPU-test, 102\nGPU-test, 103\nGPU-test, 104\n", "")
        check_gpu(gpu, own_pids={101, 102, 103, 104})
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            check_gpu(gpu, own_pids={101, 102, 103})
        run.return_value.stdout += "GPU-other, 999\n"
        check_gpu(gpu, own_pids={101, 102, 103, 104})

    def test_parallel_plan_uses_each_model_once_on_a_single_gpu(self):
        models = ["Loopie-6B", "Foil-1", "Foil-3", "Looped-MoE"]
        gpu = {"index": "0", "uuid": "GPU-test"}
        self.assertEqual(worker_plan(models, [gpu], True), [(gpu, model) for model in models])
        self.assertEqual(worker_plan(models, [gpu], False), [(gpu, None)])

    def test_four_model_driver_reaches_four_concurrent_children_without_gpu(self):
        """Exercise actual thread coordination with simulated child processes/NVML."""
        from loop_moe.expanded import save_json
        from loop_moe.trace import read_jsonl
        live = set()
        spawned = []
        peak = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = ["one", "two", "three", "four"]
            checkpoint = {"repo": "test/model", "revision": "fixed", "architecture": "foil", "dtype": "bfloat16"}
            config = {"models": models, "checkpoints": {name: checkpoint for name in models},
                      "prompts": "prompts.jsonl", "shard_size": 24, "max_new_tokens": 512}
            save_json(root / "config.json", config)
            (root / "prompts.jsonl").write_text('{"id":"a","category":"text","text":"A prompt"}\n')
            gpu = {"index": "0", "uuid": "GPU-test", "memory_mib": 0}

            class Child:
                def __init__(self, command, **kwargs):
                    self.pid = 101 + len(spawned)
                    self.returncode = None
                    model = command[command.index("--model") + 1]
                    prompt_path = Path(command[command.index("--prompts") + 1])
                    out_dir = Path(command[command.index("--out-dir") + 1]) / model
                    out_dir.mkdir(parents=True)
                    prompts = list(read_jsonl(prompt_path))
                    with gzip.open(out_dir / "trace.jsonl.gz", "wt") as stream:
                        write_line(stream, dict(checkpoint, model=model, schema_version=SCHEMA_VERSION,
                                                trace_scope=TRACE_SCOPE, request_count=len(prompts),
                                                prompts_sha256=sha256(prompt_path), max_new_tokens=512))
                        for prompt in prompts:
                            write_line(stream, {"request_id": prompt["id"]})
                    spawned.append(model)
                    live.add(self.pid)
                    peak.append(len(live))

                def poll(self):
                    if len(spawned) == 4:
                        live.discard(self.pid)
                        self.returncode = 0
                    return self.returncode

                def terminate(self):
                    self.returncode = -15
                    live.discard(self.pid)

                def wait(self, **kwargs):
                    return self.returncode

            def nvml(command, **kwargs):
                self.assertIn("--query-compute-apps=gpu_uuid,pid", command)
                return subprocess.CompletedProcess(command, 0,
                    "".join(f"GPU-test, {pid + 10000}\n" for pid in sorted(live)), "")

            args = SimpleNamespace(config=root / "config.json", out_dir=root / "out",
                                   gpus=["0"], parallel_models=True, exclusive_gpu=True, preflight_only=False)
            with patch("loop_moe.expanded.preflight", return_value=[]), \
                    patch("loop_moe.expanded.gpu_inventory", return_value=[gpu]), \
                    patch("loop_moe.expanded.subprocess.run", side_effect=nvml), \
                    patch("loop_moe.expanded.subprocess.Popen", side_effect=Child), \
                    patch("loop_moe.expanded.finish") as finish:
                run(args)
            status = __import__("json").loads((root / "out/status.json").read_text())
            self.assertEqual(status["phase"], "complete")
            self.assertEqual(len(status["completed_this_run"]), 4)
            self.assertEqual(max(peak), 4)
            self.assertCountEqual(spawned, models)
            self.assertFalse(live)
            finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
