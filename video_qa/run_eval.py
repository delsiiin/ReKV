"""Dataset scheduling and scoring, independent of the inference method."""
import os
import argparse
import subprocess
import multiprocessing
import shlex

from video_qa.sampling import load_samples, nonnegative_int
from video_qa.model_paths import MODEL_PATHS
from video_qa.methods import METHODS, get_method, validate_arguments


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = shlex.join(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env, check=True)


def get_eval_settings(args, dataset, anno_path):
    save_dir = get_method(getattr(args, 'method', 'rekv')).result_dir(args, dataset)
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


DATASETS = {
    'mlvu': ('data/mlvu/dev_debug_mc.json', 'offline', 'multiple_choice'),
    'qaego4d': ('data/qaego4d/test_mc.json', 'offline', 'multiple_choice'),
    'egoschema': ('data/egoschema/full.json', 'offline', 'egoschema'),
    'activitynet_qa': ('data/activitynet_qa/test.json', 'offline', 'open_ended'),
    'rvs_ego': ('data/rvs/ego/ego4d_oe.json', 'stream', 'open_ended'),
    'rvs_movie': ('data/rvs/movie/movienet_oe.json', 'stream', 'open_ended'),
    'cgbench': ('data/cgbench/full_mc.json', 'offline', 'multiple_choice'),
}


def evaluate(args, dataset):
    anno_path, mode, scorer = DATASETS[dataset]
    method = get_method(getattr(args, 'method', 'rekv'))
    num_chunks, save_dir = get_eval_settings(args, dataset, anno_path)
    if not args.only_eval:
        processes = []
        for index in range(num_chunks):
            command = ['python', '-m', method.solver(mode),
                       '--model', args.model,
                       '--sample_fps', str(args.sample_fps),
                       '--save_dir', save_dir, '--anno_path', anno_path,
                       '--debug', str(args.debug), '--max_samples', str(args.max_samples),
                       '--num_chunks', str(num_chunks), '--chunk_idx', str(index)]
            command.extend(method.worker_arguments(args))
            process = multiprocessing.Process(target=exec,
                args=(command, True, method.devices(args, index)))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f'{method.name} workers failed; results were not merged or scored')
        directory = shlex.quote(save_dir)
        exec(f'> {directory}/results.csv')
        if scorer == 'open_ended':
            exec(f'rm -rf {directory}/tmp')
        for index in range(num_chunks):
            part = f'{directory}/{num_chunks}_{index}.csv'
            if index == 0:
                exec(f'head -n 1 {part} > {directory}/results.csv')
            exec(f'tail -n +2 {part} >> {directory}/results.csv')
            exec(f'rm {part}')
    directory = shlex.quote(save_dir)
    if scorer == 'open_ended':
        exec(f'python video_qa/eval/eval_open_ended.py --pred_path {directory}/results.csv '
             f'--output_dir {directory}/tmp --output_json {directory}/results.json')
    else:
        exec(f'python video_qa/eval/eval_{scorer}.py --save_dir {directory}')


def eval_mlvu(args):
    return evaluate(args, 'mlvu')


def eval_qaego4d(args):
    return evaluate(args, 'qaego4d')


def eval_egoschema(args):
    return evaluate(args, 'egoschema')


def eval_activitynet_qa(args):
    return evaluate(args, 'activitynet_qa')


def eval_rvs_ego(args):
    return evaluate(args, 'rvs_ego')


def eval_rvs_movie(args):
    return evaluate(args, 'rvs_movie')


def eval_cgbench(args):
    return evaluate(args, 'cgbench')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, default='rekv')
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=MODEL_PATHS)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--num_chunks", '--num-chunks', type=int, default=1)
    parser.add_argument("--max_samples", '--max-samples', type=nonnegative_int, default=0,
                        help="Maximum video samples across all chunks; 0 uses the full dataset.")
    parser.add_argument("--only_eval", '--only-eval', action="store_true")
    parser.add_argument("--sample_fps", '--sample-fps', type=float, default=1)
    parser.add_argument("--debug", type=str, default='false')
    for method in METHODS.values():
        method.add_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args, get_method(args.method))
    except ValueError as error:
        parser.error(str(error))
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
        try:
            func_dic[args.dataset](args)
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
