import os
import argparse
import subprocess
import multiprocessing

from video_qa.sampling import load_samples, nonnegative_int
from video_qa.model_paths import MODEL_PATHS, SUPRKV_MODELS
from video_qa.suprkv_stream_vqa import add_suprkv_arguments, config_overrides


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)


def get_eval_settings(args, dataset, anno_path):
    save_dir = f"results/{args.model}/{dataset}/{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'method', 'rekv') == 'suprkv':
        budget = getattr(args, 'budget_tokens', None) or 4096
        save_dir = f"results/suprkv/{args.model}/{dataset}/{args.retrieve_size}-{args.sample_fps}/budget_{budget}"
    if args.max_samples:
        save_dir += f"/samples_{args.max_samples}"
    # Scoring existing results must not require the source annotations.
    if args.only_eval:
        return args.num_chunks, save_dir
    if args.num_chunks < 1:
        raise ValueError("num_chunks must be positive")
    samples = load_samples(anno_path, args.max_samples)
    if not samples:
        raise ValueError(f"No samples found in {anno_path}")
    num_chunks = min(args.num_chunks, len(samples))
    print(f"Selected {len(samples)} video samples across {num_chunks} chunks")
    return num_chunks, save_dir


def eval_mlvu(args):
    anno_path = "data/mlvu/dev_debug_mc.json"
    num_chunks, save_dir = get_eval_settings(args, "mlvu", anno_path)
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d(args):
    anno_path = "data/qaego4d/test_mc.json"
    num_chunks, save_dir = get_eval_settings(args, "qaego4d", anno_path)
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_egoschema(args):
    anno_path = "data/egoschema/full.json"
    num_chunks, save_dir = get_eval_settings(args, "egoschema", anno_path)
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")

def eval_activitynet_qa(args):
    anno_path = "data/activitynet_qa/test.json"
    num_chunks, save_dir = get_eval_settings(args, "activitynet_qa", anno_path)
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_ego(args):
    anno_path = "data/rvs/ego/ego4d_oe.json"
    num_chunks, save_dir = get_eval_settings(args, "rvs_ego", anno_path)
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_movie(args):
    anno_path = "data/rvs/movie/movienet_oe.json"
    num_chunks, save_dir = get_eval_settings(args, "rvs_movie", anno_path)
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_cgbench(args):
    anno_path = "data/cgbench/full_mc.json"
    num_chunks, save_dir = get_eval_settings(args, "cgbench", anno_path)
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--max_samples", str(args.max_samples),
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['rekv', 'suprkv'], default='rekv')
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=MODEL_PATHS)
    parser.add_argument("--dataset", type=str, required=True, choices=['mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench'])
    parser.add_argument("--num_chunks", '--num-chunks', type=int, default=1)
    parser.add_argument("--max_samples", '--max-samples', type=nonnegative_int, default=0,
                        help="Maximum video samples across all chunks; 0 uses the full dataset.")
    parser.add_argument("--only_eval", '--only-eval', action="store_true")
    parser.add_argument("--sample_fps", '--sample-fps', type=float, default=1)
    parser.add_argument("--n_local", type=int, default=None,
                        help='ReKV local attention window (default: 15000); not a SuPr-KV cache budget.')
    parser.add_argument("--retrieve_size", '--retrieve-size', type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument('--anno_path', '--anno-path', help='SuPr-KV annotation override')
    parser.add_argument('--save_dir', '--save-dir', help='SuPr-KV output directory override')
    parser.add_argument('--skip_eval', '--skip-eval', action='store_true',
                        help='SuPr-KV: generate and merge predictions without running the scorer')
    add_suprkv_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.method == 'suprkv':
        if args.model not in SUPRKV_MODELS:
            parser.error('SuPr-KV supports only LLaVA-OneVision Qwen2 models')
        if args.n_local is not None:
            parser.error('--n_local is ReKV-specific; configure SuPr-KV with --budget_tokens and --recent_frames')
        from video_qa.suprkv_stream_vqa import launch_evaluation
        try:
            launch_evaluation(args)
        except ValueError as error:
            parser.error(str(error))
        return
    if config_overrides(args) or args.packed_attention or args.anno_path or args.save_dir or args.skip_eval:
        parser.error('SuPr-KV options require --method suprkv')
    if args.n_local is None:
        args.n_local = 15000
    func_dic = {
        'mlvu': eval_mlvu,
        'qaego4d': eval_qaego4d,
        'egoschema': eval_egoschema,
        'activitynet_qa': eval_activitynet_qa,
        'rvs_ego': eval_rvs_ego,
        'rvs_movie': eval_rvs_movie,
        'cgbench': eval_cgbench,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)


if __name__ == "__main__":
    main()
