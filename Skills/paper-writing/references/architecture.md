# Paper architecture

## Abstract

Build one continuous argument:

1. Name the precise problem and the attractive baseline.
2. State the baseline's concrete failure.
3. Introduce the new mathematical object or algorithm in one sentence.
4. Explain its mechanism and computational cost.
5. State the main theorem with its assumptions or scope.
6. State the experimental range and principal observed result.
7. Include an important limitation when it changes interpretation.

Keep notation light. Include a symbol only if it carries the paper's central idea. Do not list every benchmark or theorem.

## Introduction

Start from the problem formulation, not a history lesson. A strong order for this author's papers is:

- target object and computational goal;
- why the natural method is insufficient;
- precise error metric or failure diagnostic;
- the proposed construction and its mechanism;
- relation to the nearest methods;
- contributions in prose or a locally consistent list;
- organization.

Present enough mathematics to make the contribution concrete. A defining equation for the loss, estimator, dynamics, or error decomposition often belongs in the introduction. Explain every term and state what is unchanged (minimizer, architecture, inference cost) when that is a central advantage.

## Related work

Organize by technical axis, not a citation chronology. Compare assumptions, information requirements, target geometry, guarantees, computational cost, and failure modes. Explain the exact distinction after each citation cluster. Avoid claiming that prior work “cannot” do something unless the cited method and assumptions establish that.

## Preliminaries and assumptions

Fix notation before assumptions, and assumptions before the first theorem that uses them. Group assumptions by role: geometry/regularity, stochastic oracle, moment condition, numerical stability. After each group, explain where it enters and whether it is standard, restrictive, or only technical.

Define the algorithm with enough precision to support the analysis. Establish a notation system for exact, numerical, coupled, and reference processes before error decompositions begin.

## Theory sections

Lead with the headline theorem or a roadmap to it. A useful progression is:

- define the analytical object;
- state a contractivity, regularity, or moment lemma;
- derive the local decomposition;
- state the global theorem with full parameter dependence;
- interpret each term and its convergence order;
- defer long technical estimates to the supplement while retaining the proof mechanism in the main text.

State the theorem before burying the reader in proof machinery whenever possible. After the theorem, translate the bound into error sources, stability conditions, complexity, or limiting regimes.

## Method sections

First fix distributions, potentials, maps, and direction. Then:

1. Write the objective in the variables actually implemented.
2. Explain every term's role.
3. Identify how each expectation or sample set is obtained.
4. Present algorithms in dependency order.
5. Discuss differentiability, stop-gradients, normalization, and numerical stability.
6. State training and inference costs separately.
7. Give default hyperparameters only when supported by evidence.
8. Name failure modes and remedies without confusing a workaround with a guarantee.

## Experiments

Open with the scientific questions and benchmark families. Design each benchmark to isolate a mechanism: mode discovery, calibration, dimensional scaling, stiffness, singularity, or variance reduction. State target, method, architecture, budget, and metrics before interpreting results.

Within each result block use:

- setup or isolated difficulty;
- observed comparison;
- diagnostic evidence;
- mechanism-level interpretation;
- qualification.

Aggregate patterns after individual cases. Do not force the reader to infer the conclusion from a table. Conversely, do not repeat every number in prose.

## Conclusions and limitations

Restate the problem and the achieved mechanism, not the paper outline. Summarize theoretical and computational contributions at the level established by evidence. Then state where gains are small, where the construction fails, and which unresolved issue is structural rather than an implementation detail. Future work should follow directly from these limitations.

## Supplement

Keep the main paper self-contained at the level of definitions, headline results, method, and experimental conclusions. Put complete proofs, secondary benchmarks, extended protocols, sensitivity tests, and molecular/system details in the supplement. Use synchronized notation and cross-references; do not let the supplement become a second inconsistent paper.
