# Rigor and evidence checks

## Theorems and proofs

For every theorem verify:

- all symbols and spaces are defined;
- assumptions are explicit and sufficient;
- quantifiers and initial conditions are present;
- constants state their dependencies and independence;
- parameter ranges and step-size restrictions are visible;
- dimensions and norms are consistent;
- the prose interpretation matches every term in the bound.

Structure a long proof by naming the controlling decomposition first. Separate martingale terms, deterministic bias, local discretization error, and remainder terms before estimating them. State which earlier lemma controls each term. When summing local bounds, show where independence, conditional centering, contractivity, Cauchy--Schwarz, or a discrete Grönwall estimate enters.

Do not disguise an informal equality as a theorem. Label it a conjecture or heuristic and state the missing argument.

## SciML method claims

Check map orientation, density transformations, Jacobian signs, normalization constants, detached quantities, and sampling measures. Distinguish:

- the population objective from its batch estimator;
- exact target samples from biased MCMC/AIS surrogates;
- continuous gradient-flow analysis from discrete optimizer behavior;
- proposal quality from reweighted sample quality;
- asymptotic correctness from finite-budget mixing;
- theoretical cost from actual memory and wall time.

A theorem about an idealized density flow does not imply an optimizer convergence rate. Say explicitly what the idealization serves as evidence for.

## Numerical studies

Each comparison should specify or point to:

- target and source distributions;
- dimensions and boundary conditions;
- architecture and parameter count when relevant;
- optimizer, learning rate, steps, batch/pool size;
- MCMC/integrator step size and iteration count;
- number of independent runs or a clear statement that the result is a single run;
- randomness and uncertainty summary;
- compute hardware when runtime/memory is claimed;
- metric definition and normalization;
- common budget or an explanation of unequal budgets.

Use complementary metrics. A density-fit metric may miss mode coverage; a coverage metric may miss calibration; final error may hide transient instability. Where a metric has a blind spot, construct or report a diagnostic that detects it.

Do not call a visual difference significant. Reserve “statistically significant” for an actual statistical test. Do not bold a winner that fails a required constraint; for example, select among full-coverage methods if coverage is part of correctness.

## Figures and tables

A caption should make the object understandable without searching the body. State rows, columns, methods, metrics, normalization, and key protocol differences. Keep detailed interpretation in the body.

Check that axes have quantities and units, legends agree with prose, colors/markers remain distinguishable, table precision reflects uncertainty, and bolding follows a declared rule. Ensure every figure/table is cited before or near its appearance and the text states the conclusion it supports.

## Citations

For every new citation:

1. Confirm the key exists in the active `.bib` file.
2. Inspect the paper or reliable source.
3. Match the citation to the exact claim: definition, theorem, method, benchmark, or historical attribution.
4. Avoid citation bundles whose papers support different parts of one sentence.
5. Do not use a secondary source for a priority claim when the primary source is available.

## Cross-paper consistency

Search the main text, supplement, captions, and abstract for every headline number and named setting. Confirm one source of truth for:

- theorem rates and assumptions;
- dimensions and sample counts;
- default hyperparameters;
- method names and abbreviations;
- table values quoted in prose;
- claims of best/tied-best behavior;
- limitations.

## Read-only validation

Inspect publication trees without modifying them unless edits are requested. To compile or lint, copy the paper and required assets to a temporary directory, run the existing toolchain there, inspect the resulting PDF, then remove the copy. Before handoff, check the original working tree or file timestamps to confirm validation created no source-side artifacts.
