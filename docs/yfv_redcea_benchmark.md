# YFV RedCEA Benchmark

This benchmark runs donor-matched RedCEA on six Yellow Fever vaccination pairs, annotates LLW/VDJdb exact matches as partial validation, measures cross-donor overlap before and after RedCEA filtering, and writes manuscript-facing summaries plus figures.

The config lives at `configs/yfv_redcea_benchmark.yaml`. It uses JSON-compatible YAML so the scripts can parse it without an extra YAML dependency. Update the donor file paths, LLW/VDJdb path, and column names there before running on the server.

Run each numbered step from the repo root:

```bash
python scripts/yfv/1_prepare_yfv_inputs.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
python scripts/yfv/2_run_yfv_redcea_grid.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
python scripts/yfv/3_annotate_yfv_llw.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
python scripts/yfv/4_summarize_yfv_benchmark.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
python scripts/yfv/5_compute_yfv_overlap.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
python scripts/yfv/6_plot_yfv_benchmark.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
python scripts/yfv/7_write_yfv_benchmark_summary.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
```

To run everything in one command:

```bash
python scripts/yfv/0_run_all_yfv_benchmark.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark
```

For a dry run that checks config structure, donor path definitions, output directories, existing-file columns, and grid validity:

```bash
python scripts/yfv/0_run_all_yfv_benchmark.py --config configs/yfv_redcea_benchmark.yaml --outdir results/yfv_redcea_benchmark --dry-run
```

Outputs are written under `results/yfv_redcea_benchmark/`:

- `runs/` for RedCEA run directories and assignment parquets.
- `summaries/` for donor, cluster, LLW, overlap, config, and best-config TSVs.
- `figures/` for the four required benchmark figures.
- `logs/` for manifests, normalized AIRR inputs, LLW match tables, and run metadata.
- `YFV_BENCHMARK_SUMMARY.md` for the top-level explanation of what was run and how to read the outputs.

Interpret LLW recovery as a sanity check only. No LLW match does not imply a false positive. Prefer configs that balance volcano behavior, overlap gain, retained sample fraction, and fragmentation rather than optimizing LLW recovery alone.
