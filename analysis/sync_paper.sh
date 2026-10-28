#!/bin/sh
# Copy the tables and figures produced by the notebooks into the paper repo (run the notebooks first)
cd "$(dirname "$0")" && cp _tables/rq0.tex _tables/rq1.tex _tables/rq2.tex _tables/rq3.tex ../_paper/tables/ \
  && cp _figures/rq0_ablation.pdf _figures/tradeoff_methods.pdf _figures/semantic_classes.pdf _figures/spec_asr_013.jpg ../_paper/data/figs/
