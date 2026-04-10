import multiprocessing as mp

from redcea.arguments import get_arguments_enrich
from redcea.config import PipelineConfig
from redcea.pipeline import run_redcea_pipeline


def main(argv=None):
    args = get_arguments_enrich(argv)
    return run_redcea_pipeline(PipelineConfig.from_args(args))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
