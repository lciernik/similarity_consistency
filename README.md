# Diverse Priors

Can we combine representations of different models to improve label efficiency and robustness?

## Getting Started

To setup the feature extraction and benchmark:

```bash
cd benchmark
pip install .
```

Scripts to run feature extraction and evaluation of combined models are found in `benchmark/scripts`.
To check if you set up everything correctly, you can run `./benchmark/scripts/test_scripts/test_single.sh`.

## Sampling

We want to test different model selection strategies.
Let's assume we want to sample `num_samples` model sets of size `k`.
Then, we can do that with the following command.

```bash
python clip_benchmark/sample_models.py \
  --num_models <k> \
  --sampling strategies top-k random cluster one-cluster \
  --num_samples <num_samples> \
  --output_root /home/space/diverse_priors/sampling/models_<k>-samples_<num_samples>
```

This will write a json with the sampled model sets for each of the sampling strategies into
`/home/space/diverse_priors/sampling/models_<k>-samples_<num_samples>`.

To run sampling for multiple model set sizes, we can use `benchmark/scripts/run_sampling.py` \
which will automatically trigger the necessary jobs.

