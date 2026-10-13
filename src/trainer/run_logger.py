import os
import json
import datetime
import platform
import torch
import numpy as np

from ..helper import save_audio
from ..metrics import is_success, wer


class RunLogger:
    def __init__(self, active_objectives, tts_model, asr_model, vector_manipulator, device: str):
        """
        Initializes the logger with specific run results.
        """
        self.active_objectives = active_objectives
        self.tts_model = tts_model
        self.asr_model = asr_model
        self.vector_manipulator = vector_manipulator
        self.device = device

        # Initialize Directory
        self.folder_path = None

    def save_audios(self, audio_gt, audio_target, audio_best_mixed):

        save_audio(audio_gt, os.path.join(self.folder_path, "ground_truth.wav"))

        if audio_target is not None:
            save_audio(audio_target, os.path.join(self.folder_path, "target.wav"))

        save_audio(audio_best_mixed, os.path.join(self.folder_path, "best_mixed.wav"))

    def save_archive_history(self, archive_history: list):
        """Archive fitness after every generation (archive_history.json), for success-vs-query-budget analysis."""
        obj_names = [obj.name for obj in self.active_objectives]
        history = {
            "objectives": obj_names,
            "archive_per_generation": [np.round(np.asarray(a, dtype=np.float64), 6).tolist() for a in archive_history],
        }
        with open(os.path.join(self.folder_path, "archive_history.json"), "w") as f:
            json.dump(history, f)

    def select_best_candidate(self, candidates, thresholds=None, text_gt=None, intent_first=False):
        if not candidates:
            raise ValueError("Candidate list is empty.")

        # Intent matching: select by the intent objective, not the ASR test-case criterion. Among candidates that reach the
        # target intent (intent fitness within its threshold) the best naturalness wins; if none does, the closest one.
        intent_cols = [i for i, obj in enumerate(self.active_objectives) if obj.name.startswith("INTENT_")]
        if intent_first and intent_cols:
            f = np.array([c.fitness for c in candidates], dtype=np.float64)
            col = intent_cols[0]
            limit = (thresholds or {}).get(self.active_objectives[col], 0.5)
            hits = np.where(f[:, col] <= limit)[0]
            if len(hits):
                others = [i for i in range(f.shape[1]) if i != col]
                best = hits[np.argmin(f[hits][:, others].sum(axis=1))] if others else hits[0]
                print(f"[Log] {len(hits)}/{len(candidates)} candidate(s) reach the target intent — selected the most natural.")
            else:
                best = int(np.argmin(f[:, col]))
                print("[Log] No candidate reaches the target intent — selected the closest.")
            selected = candidates[best]
            print(f"[Log] Selected Candidate Fitness: {selected.fitness.tolist()}")
            return selected

        # Step 0: Prefer candidates that already are test cases (the transcription differs from the ground truth
        # text, the success criterion of the evaluation). Uses the stored transcriptions, no extra ASR queries.
        if text_gt is not None:
            found = [c for c in candidates if c.data is not None and len(c.data) > 1 and c.data[1] is not None
                     and is_success(wer(text_gt, c.data[1]))]
            if found:
                print(f"[Log] {len(found)}/{len(candidates)} Pareto candidate(s) are test cases — restricting selection.")
                candidates = found
            else:
                print("[Log] No Pareto candidate is a test case — using the full Pareto front.")

        f = np.array([c.fitness for c in candidates])
        indices = np.arange(len(candidates))
        print(f"\n[Log] Candidates on Pareto Front: {len(candidates)}")

        # Step 1: Restrict to candidates that meet all thresholds (if any do).
        # Without this filter, a candidate with near-zero value on one objective but
        # violating the other threshold can win via [min,max] range distortion.
        if thresholds:
            meets_all = np.ones(len(candidates), dtype=bool)
            for i, obj in enumerate(self.active_objectives):
                if obj in thresholds:
                    meets_all &= (f[:, i] <= thresholds[obj])
            if np.any(meets_all):
                print(f"[Log] {np.sum(meets_all)} candidate(s) meet all thresholds — restricting selection.")
                indices = indices[meets_all]
                f = f[meets_all]
            else:
                print(f"[Log] No candidate meets all thresholds — using full Pareto front.")

        # Step 2: Scale each objective dimension.
        # Positive threshold → divide by threshold (threshold maps to 1.0, giving it
        # proportional weight: tighter threshold = higher weight).
        # Zero/missing threshold → divide by observed max (normalises to [0, 1]).
        working_f = f.copy().astype(np.float64)
        for i, obj in enumerate(self.active_objectives):
            t = thresholds.get(obj, 0.0) if thresholds else 0.0
            if t > 0.0:
                working_f[:, i] /= t
            else:
                col_max = working_f[:, i].max()
                working_f[:, i] /= col_max if col_max > 0 else 1.0

        # Step 3: Pick the candidate closest to the origin using the L3 norm.
        # L3 penalises large coordinates more than L2, rewarding both smallness and
        # balance: (0.3, 0.3) beats (0.1, 0.9) because 0.3³+0.3³ < 0.1³+0.9³.
        # No [min,max] re-normalisation here — that step would be distorted by
        # extreme Pareto corners (e.g. IV≈0 solutions) and undo the threshold scaling.
        distances = np.linalg.norm(working_f, ord=3, axis=1)
        best_local_idx = np.argmin(distances)
        selected = candidates[indices[best_local_idx]]
        print(f"[Log] Selected Candidate Fitness: {selected.fitness.tolist()}")
        return selected

    def run_final_inference(self, best_candidate, batch_size: int = 1):
        audio_best = best_candidate.data[0].unsqueeze(0).to(self.device)

        # Use the transcription stored during optimization (same batch context → deterministic).
        # Fall back to re-inference only for candidates created before this fix.
        if best_candidate.data is not None and len(best_candidate.data) > 1 and best_candidate.data[1]:
            return audio_best, best_candidate.data[1]

        asr_model = self.asr_model.module if isinstance(self.asr_model, torch.nn.DataParallel) else self.asr_model
        audio_batch = audio_best.expand(batch_size, -1)
        asr_texts, _ = asr_model.inference(audio_batch)
        asr_text = asr_texts[0] if isinstance(asr_texts, list) else asr_texts

        return audio_best, asr_text

    def setup_multi_sentence_directory(self, sentence_id: int, run_id: int, run_timestamp: str, base_path: str = "outputs/results"):
        """Creates the structured results folder for a Harvard sentences run."""
        folder_path = os.path.join(base_path, run_timestamp, f"sentence_{sentence_id:03d}", f"run_{run_id}")
        os.makedirs(folder_path, exist_ok=True)
        print(f"[Log] Results directory initialized: {folder_path}")
        return folder_path

    def save_json_summary(
        self,
        text_best: str,
        best_candidate,
        optimizer,
        config_data,
        generation_count: int,
        elapsed_time_total: float,
        num_generations: int = None,
        sentence_id: int = None,
        run_id: int = None,
        run_timestamp: str = None,
        generation_found: int = None,
        seed_target: bool = False,
        target_asr_text: str = "",
        gt_rms: float = None,
        target_rms: float = None,
        gt_asr_text: str = "",
        optimization_time_seconds: float = None,
        seed: int = None,
        method: str = None,
        dataset: str = None,
    ) -> dict:
        gpu_info = "CPU Only"
        if torch.cuda.is_available():
            device_index = torch.device(self.device).index or 0
            vram = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
            gpu_info = f"{torch.cuda.get_device_name(device_index)} ({vram:.2f} GB VRAM)"

        avg_per_gen = elapsed_time_total / generation_count if generation_count > 0 else 0
        fitness_names = [obj.name for obj in self.active_objectives]
        fitness_dict = dict(zip(fitness_names, [float(v) for v in best_candidate.fitness]))
        thresholds = {k.name: v for k, v in config_data.thresholds.items()} if config_data.thresholds else {}

        summary = {
            "metadata": {
                "attack_method": method,
                "dataset": dataset,
                "run_timestamp": run_timestamp,
                "sentence_id": sentence_id,
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "hardware": gpu_info,
                "os": f"{platform.system()} {platform.release()}",
                "cpu": platform.processor(),
            },
            "text_data": {
                "ground_truth_text": config_data.text_gt,
                "gt_transcription": gt_asr_text,
                "target_text": config_data.text_target if config_data.mode.name == "TARGETED" else target_asr_text,
                "asr_transcription": text_best,
                "target_intent": config_data.target_intent,
            },
            "objectives": {
                # Optimizer fitness of the selected candidate and the objective thresholds
                "fitness_scores": fitness_dict,
                "thresholds": thresholds,
            },
            "efficiency_metrics": {
                "generation_count": generation_count,
                # Attacked-ASR queries (one per evaluated individual)
                "queries": generation_count * config_data.pop_size,
                # End-to-end wall clock from attack start to selected result
                "elapsed_time_seconds": round(elapsed_time_total, 2),
                "avg_time_per_generation": round(avg_per_gen, 2),
                "optimization_time_seconds": round(optimization_time_seconds, 2) if optimization_time_seconds is not None else None,
            },
            "algorithm_parameters": {
                "attack_mode": config_data.mode.name,
                "asr_model": config_data.asr_model_name,
                "seed": seed,
                "objectives": fitness_names,
                "pop_size": config_data.pop_size,
                "num_generations": num_generations,
                "rank_text": config_data.rank_text,
                "rank_style": config_data.rank_style,
                "adaptive_scale": config_data.adaptive_scale,
                "basis": config_data.basis,
                "text_dct": config_data.text_dct,
                "iv_scalar": config_data.iv_scalar,
                "num_rms_candidates": getattr(config_data, "num_rms_candidates", 1),
                "seed_target": seed_target,
                "gt_rms": round(gt_rms, 6) if gt_rms is not None else None,
                "target_rms": round(target_rms, 6) if target_rms is not None else None,
            },
            "final_solution": {
                "generation_found": generation_found,
                # Perturbation scale of the selected candidate (adaptive: its last gene, else the fixed iv_scalar)
                "scale": self.vector_manipulator.scale_of(best_candidate.solution) if self.vector_manipulator is not None else None,
                "fitness_scores": fitness_dict,
            },
            "file_paths": {
                "best_mixed_audio": "best_mixed.wav",
                "ground_truth_audio": "ground_truth.wav",
            },
        }

        save_path = os.path.join(self.folder_path, "run_summary.json")
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Log] Run summary saved (sentence={sentence_id}, run={run_id})")
        return summary

    def save_results_run(
        self,
        optimizer,
        fitness_data: list,
        archive_data: list,
        generation_count: int,
        elapsed_time_total: float,
        audio_gt,
        audio_target,
        config_data,
        folder_path: str,
        num_generations: int = None,
        sentence_id: int = None,
        run_id: int = None,
        run_timestamp: str = None,
        generation_found: int = None,
        seed_target: bool = False,
        gt_rms: float = None,
        target_rms: float = None,
        optimization_time_seconds: float = None,
        seed: int = None,
        method: str = None,
        dataset: str = None,
    ) -> dict:
        """elapsed_time_total: end-to-end wall clock of the attack (start until the optimizer returned)."""
        os.makedirs(folder_path, exist_ok=True)
        self.folder_path = folder_path

        best_candidate = self.select_best_candidate(optimizer.best_candidates, config_data.thresholds, config_data.text_gt,
                                                 intent_first=config_data.intent_selection)
        audio_best, text_best = self.run_final_inference(best_candidate, batch_size=min(config_data.batch_size, config_data.pop_size))

        # Transcribe target audio for non-TARGETED modes (e.g. NOISE_UNTARGETED)
        target_asr_text = ""
        if config_data.mode.name != "TARGETED" and audio_target is not None:
            asr_model = self.asr_model.module if isinstance(self.asr_model, torch.nn.DataParallel) else self.asr_model
            target_audio_tensor = audio_target.detach().cpu().unsqueeze(0) if audio_target.dim() == 1 else audio_target.detach().cpu()
            target_asr_texts, _ = asr_model.inference(target_audio_tensor.to(self.device))
            target_asr_text = target_asr_texts[0] if target_asr_texts else ""

        # Transcribe ground truth audio
        asr_model = self.asr_model.module if isinstance(self.asr_model, torch.nn.DataParallel) else self.asr_model
        gt_audio_tensor = audio_gt.detach().cpu().unsqueeze(0) if audio_gt.dim() == 1 else audio_gt.detach().cpu()
        gt_asr_texts, _ = asr_model.inference(gt_audio_tensor.to(self.device))
        gt_asr_text = gt_asr_texts[0] if gt_asr_texts else ""

        self.save_audios(audio_gt, audio_target, audio_best)
        self.save_archive_history(archive_data)
        summary = self.save_json_summary(text_best, best_candidate, optimizer, config_data, generation_count, elapsed_time_total, num_generations, sentence_id, run_id, run_timestamp, generation_found=generation_found, seed_target=seed_target, target_asr_text=target_asr_text, gt_rms=gt_rms, target_rms=target_rms, gt_asr_text=gt_asr_text, optimization_time_seconds=optimization_time_seconds, seed=seed, method=method, dataset=dataset)

        return summary

    # =========================================================================
    # Aggregation
    # =========================================================================

    @staticmethod
    def _flatten_summary(summary: dict) -> dict:
        row = {}

        meta = summary.get("metadata", {})
        row["attack_method"] = meta.get("attack_method")
        row["run_timestamp"] = meta.get("run_timestamp")
        row["sentence_id"] = meta.get("sentence_id")
        row["run_id"] = meta.get("run_id")
        row["timestamp"] = meta.get("timestamp")
        row["hardware"] = meta.get("hardware")

        text = summary.get("text_data", {})
        row["ground_truth_text"] = text.get("ground_truth_text")
        row["gt_transcription"] = text.get("gt_transcription")
        row["target_text"] = text.get("target_text")
        row["asr_transcription"] = text.get("asr_transcription")

        params = summary.get("algorithm_parameters", {})
        row["asr_model"] = params.get("asr_model")
        row["seed"] = params.get("seed")

        objectives = summary.get("objectives", {})
        for obj, threshold in objectives.get("thresholds", {}).items():
            row[f"threshold_{obj}"] = threshold
        for obj, score in objectives.get("fitness_scores", {}).items():
            row[f"score_{obj}"] = score

        eff = summary.get("efficiency_metrics", {})
        row["generation_count"] = eff.get("generation_count")
        row["elapsed_time_seconds"] = eff.get("elapsed_time_seconds")
        row["avg_time_per_generation"] = eff.get("avg_time_per_generation")
        row["optimization_time_seconds"] = eff.get("optimization_time_seconds")

        algo = summary.get("algorithm_parameters", {})
        row["attack_mode"] = algo.get("attack_mode")
        row["objectives"] = ",".join(algo.get("objectives", []))
        row["pop_size"] = algo.get("pop_size")
        row["num_generations"] = algo.get("num_generations")
        row["rank_text"] = algo.get("rank_text")
        row["rank_style"] = algo.get("rank_style")
        row["iv_scalar"] = algo.get("iv_scalar")

        sol = summary.get("final_solution", {})
        row["generation_found"] = sol.get("generation_found")
        row["dataset"] = meta.get("dataset")
        row["queries"] = eff.get("queries")

        return row

    @staticmethod
    def aggregate_results(summaries: list, output_dir: str = "outputs"):
        """
        Aggregate a list of run_summary dicts into all_results.json and all_results.csv.

        Args:
            summaries: List of dicts as returned by save_results_run.
            output_dir: Directory to write output files into.
        """
        import csv as _csv

        if not summaries:
            print("[Aggregate] No summaries to aggregate.")
            return []

        all_rows = [RunLogger._flatten_summary(s) for s in summaries]

        os.makedirs(output_dir, exist_ok=True)

        json_out = os.path.join(output_dir, "all_results.json")
        with open(json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"[Aggregate] Saved {json_out}")

        fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
        csv_out = os.path.join(output_dir, "all_results.csv")
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[Aggregate] Saved {csv_out}")

        total = len(all_rows)
        sentences = sorted(set(r["sentence_id"] for r in all_rows if r["sentence_id"] is not None))
        print(f"[Aggregate] Total runs: {total}")
        print(f"[Aggregate] Sentences: {len(sentences)} ({min(sentences)} – {max(sentences)})")

        return all_rows
