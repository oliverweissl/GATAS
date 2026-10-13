"""
GATAS: black-box ASR test generation by NSGA-II search over a low-rank additive perturbation of the StyleTTS2
text embedding and style vectors of the sentence.

Generate reference audios first:
    python scripts/generate_reference_audios.py --dataset harvard --start 1 --end 100

Usage (see scripts/run_gatas.sh for the paper settings):
    python scripts/adversarial_gatas.py --dataset harvard --start 1 --end 100 [args]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import signal
import argparse
import torch
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from src.data.dataclass import ModelData
from src.data.datasets import DATASETS, load_sentences, load_target_intents, load_intent_rewrites
from src.helper import set_seed, run_seed, load_reference_embedding, results_dir, experiment_name, target_sentence
from src.trainer.environment_loader import EnvironmentLoader
from src.trainer.adversarial_trainer import AdversarialTrainer
from src.trainer.run_logger import RunLogger
from src.trainer.vector_manipulator import VectorManipulator, InterpolationManipulator
from src.optimizer.pymoo_optimizer import PymooOptimizer
from src.optimizer.cma_optimizer import CmaesOptimizer, MoCmaOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2



from src.models import ASR_MODEL_CHOICES, canonical_asr_model_name

def initialize_parser():
    parser = argparse.ArgumentParser(description="GATAS")
    parser.add_argument("--dataset", type=str, default="harvard", choices=DATASETS)
    parser.add_argument("--start", type=int, default=1, help="First sentence ID (1-based)")
    parser.add_argument("--end", type=int, default=100, help="Last sentence ID (1-based, inclusive)")
    parser.add_argument("--loop_count", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--iv_scalar", type=float, default=2.0, help="Perturbation size in standard deviations per coefficient")
    parser.add_argument("--adaptive_scale", action="store_true", help="Search the scale as an extra gene in [0.5, iv_scalar]")
    parser.add_argument("--basis", type=str, default="pca", choices=["pca", "jacobian"],
                        help="Perturbation directions: PCA of the embeddings, or ranked by their effect on the TTS output")
    parser.add_argument("--text_dct", type=int, default=0,
                        help="If > 0, text coefficients are this many DCT components along the phoneme axis instead of one set per phoneme")
    parser.add_argument("--algorithm", type=str, default="nsga2", choices=["nsga2", "cmaes", "mocma"],
                        help="nsga2: NSGA-II; cmaes: CMA-ES on max_i f_i / threshold_i; mocma: MO-CMA-ES")
    parser.add_argument("--rank_text", type=int, default=8, help="Coefficients per phoneme (text embedding)")
    parser.add_argument("--rank_style", type=int, default=16, help="Coefficients per style vector (acoustic, prosodic)")
    parser.add_argument("--mode", type=str, default="TARGETED", help="TARGETED: objectives compare to a target text; else untargeted")
    parser.add_argument("--method_name", type=str, default=None,
                        help="Result folder / attack_method name (default: GATAS, or GATAS_targeted in TARGETED mode)")
    parser.add_argument("--target_text", type=str, default="")
    parser.add_argument("--target_interp", action="store_true",
                        help="GATAS_llm: search the interpolation toward the LLM intent rewrite (<dataset>.intent_rewrites.txt) "
                             "in latent space instead of perturbing the seed")
    parser.add_argument("--interp_knots", type=int, default=8, help="--target_interp: alphas along the phoneme axis")
    parser.add_argument("--intent_selection", action="store_true",
                        help="Intent matching: pick the final candidate by the intent objective (reaches target, then most natural)")
    parser.add_argument("--objectives", type=str, default="PESQ=0.2, SET_OVERLAP=0.5")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (per run: seed + sentence_id + 1000 * run_id)")
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index to use (default: 0)')
    parser.add_argument('--asr_model', type=str, default='whisper-tiny', choices=ASR_MODEL_CHOICES,
                        help='ASR backend to optimize/evaluate against')
    return parser


def main():
    parser = initialize_parser()
    args = parser.parse_args()

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    method = args.method_name or ("GATAS_intent" if "INTENT_" in args.objectives else "GATAS_targeted" if args.mode == "TARGETED" else "GATAS")
    loader = EnvironmentLoader(device)
    tts_model, asr_model = loader.load_required_models(args.asr_model)
    print("Models loaded.")

    asr_model_name = canonical_asr_model_name(args.asr_model)
    sentences = load_sentences(args.dataset)
    run_timestamp = experiment_name(asr_model_name)
    print(f"Run timestamp: {run_timestamp}")
    print(f"ASR model: {args.asr_model}")
    print(f"{'='*60}")
    print(f"  mode:               {args.mode}")
    print(f"  dataset:            {args.dataset}")
    print(f"  sentences:          {args.start} → {args.end}")
    print(f"  runs per sentence:  {args.loop_count}")
    print(f"  generations:        {args.num_generations}")
    print(f"  pop_size:           {args.pop_size}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  objectives:         {args.objectives}")
    print(f"  iv_scalar:          {args.iv_scalar}")
    print(f"  rank_text/style:    {args.rank_text}/{args.rank_style}")
    print(f"  basis / text_dct:   {args.basis} / {args.text_dct}")
    print(f"  algorithm:          {args.algorithm}")
    print(f"{'='*60}")

    all_summaries = []

    try:
        for sentence_id in range(args.start, args.end + 1):
            sentence_text = sentences[sentence_id - 1]
            target_text = args.target_text if args.target_text else target_sentence(sentences, sentence_id, args.seed)
            print(f"\n{'='*60}")
            target_intent = load_target_intents(args.dataset)[sentence_id - 1] if "INTENT_" in args.objectives else None
            print(f"[Sentence {sentence_id}] {sentence_text}")
            if target_intent:
                print(f"[Target intent] {target_intent}")
            if args.mode == "TARGETED":
                print(f"[Target]     {target_text}")
            print(f"{'='*60}")

            for run_id in range(args.loop_count):
                print(f"\n  [Run {run_id + 1}/{args.loop_count}]")
                set_seed(run_seed(args.seed, sentence_id, run_id))
                attack_start = time.time()

                run_args = argparse.Namespace(
                    ground_truth_text=sentence_text,
                    target_text=target_text,
                    loop_count=1,
                    num_generations=args.num_generations,
                    pop_size=args.pop_size,
                    iv_scalar=args.iv_scalar,
                    rank_text=args.rank_text,
                    rank_style=args.rank_style,
                    adaptive_scale=args.adaptive_scale,
                    basis=args.basis,
                    text_dct=args.text_dct,
                    batch_size=args.batch_size,
                    asr_model=asr_model_name,
                    mode=args.mode,
                    objectives=args.objectives,
                )
                config_data = loader.load_configuration(run_args)
                config_data.target_intent = target_intent
                config_data.intent_selection = args.intent_selection

                tokens = tts_model.preprocess_text(config_data.text_gt)
                audio_embedding_gt = load_reference_embedding(args.dataset, sentence_id, device) \
                    or tts_model.extract_embeddings(tokens, torch.randn(1, 1, 256, device=device))
                audio_gt = tts_model.inference_on_embedding(audio_embedding_gt).flatten()
                gt_rms = audio_gt.pow(2).mean().sqrt().item()

                objectives_dict = loader.initialize_objectives(
                    active_objectives=config_data.active_objectives,
                    model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
                    text_gt=config_data.text_gt,
                    text_target=config_data.text_target,
                    mode=config_data.mode,
                    audio_gt=audio_gt,
                    target_intent=target_intent,
                )

                if args.target_interp:
                    rewrite = load_intent_rewrites(args.dataset)[sentence_id - 1]
                    print(f"[Rewrite]    {rewrite}")
                    # same style noise as the seed's reference (generate_reference_audios.py)
                    noise = torch.randn(1, 1, 256, generator=torch.Generator().manual_seed(run_seed(args.seed, sentence_id))).to(device)
                    vector_manipulator = InterpolationManipulator(tts_model, audio_embedding_gt, tts_model.preprocess_text(rewrite), noise, args.interp_knots)
                else:
                    vector_manipulator = VectorManipulator(tts_model, tokens, audio_embedding_gt, config_data)
                trainer = AdversarialTrainer(
                    tts_model, asr_model, config_data.thresholds, objectives_dict, vector_manipulator, device
                )
                logger = RunLogger(
                    config_data.active_objectives, tts_model, asr_model, vector_manipulator, device
                )

                solution_shape = (vector_manipulator.n_genes,)
                num_objectives = len(config_data.active_objectives)
                if args.algorithm == "cmaes":
                    optimizer = CmaesOptimizer(
                        bounds=(0, 1), pop_size=args.pop_size, num_objectives=num_objectives, solution_shape=solution_shape,
                        weights=[1.0 / config_data.thresholds[o] if config_data.thresholds.get(o, 0.0) > 0 else 1.0 for o in config_data.active_objectives],
                    )
                elif args.algorithm == "mocma":
                    optimizer = MoCmaOptimizer(
                        bounds=(0, 1), pop_size=args.pop_size, num_objectives=num_objectives, solution_shape=solution_shape,
                    )
                else:
                    optimizer = PymooOptimizer(
                        bounds=(0, 1),
                        algorithm=NSGA2,
                        algo_params={"pop_size": args.pop_size},
                        num_objectives=num_objectives,
                        solution_shape=solution_shape,
                    )

                fitness_data, archive_data, generation_count, optimization_time, interrupted, generation_found = \
                    trainer.run_full_iteration(optimizer, args.num_generations, args.pop_size, args.batch_size)
                elapsed_time_total = time.time() - attack_start

                if fitness_data:
                    folder_path = logger.setup_multi_sentence_directory(sentence_id, run_id, run_timestamp, base_path=os.path.dirname(results_dir(method, args.dataset, run_timestamp)))
                    summary = logger.save_results_run(
                        optimizer=optimizer,
                        fitness_data=fitness_data,
                        archive_data=archive_data,
                        generation_count=generation_count,
                        elapsed_time_total=elapsed_time_total,
                        audio_gt=audio_gt,
                        audio_target=None,
                        config_data=config_data,
                        folder_path=folder_path,
                        sentence_id=sentence_id,
                        run_id=run_id,
                        run_timestamp=run_timestamp,
                        num_generations=args.num_generations,
                        generation_found=generation_found,
                        gt_rms=gt_rms,
                        target_rms=None,
                        optimization_time_seconds=optimization_time,
                        seed=run_seed(args.seed, sentence_id, run_id),
                        method=method,
                        dataset=args.dataset,
                    )
                    all_summaries.append(summary)

                torch.cuda.empty_cache()

                if interrupted:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[!] Experiment stopped early. All completed runs have been saved.")

    finally:
        RunLogger.aggregate_results(all_summaries, output_dir=results_dir(method, args.dataset, run_timestamp))
        print("\n[Done]")


if __name__ == "__main__":
    main()
