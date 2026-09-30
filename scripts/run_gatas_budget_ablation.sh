#!/bin/bash
# Experimental budget ablation of GATAS (main budget 20 x 10 = 200 queries is run by run_gatas.sh):
#   GATAS_budget300    population 20 x 15 generations  =    300 queries (SMACK's budget)
#   GATAS_budget10000  population 100 x 100 generations = 10,000 queries (budget of the first paper version)
# Run from project root: bash scripts/run_gatas_budget_ablation.sh
POP_SIZE=20  NUM_GENERATIONS=15  METHOD_NAME=GATAS_budget300   bash scripts/run_gatas.sh
POP_SIZE=100 NUM_GENERATIONS=100 METHOD_NAME=GATAS_budget10000 bash scripts/run_gatas.sh
