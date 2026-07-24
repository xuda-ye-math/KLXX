# Author voice derived from the publication corpus

## Core register

The voice is mathematical, direct, and explanatory. It does not perform sophistication; it makes the mechanism visible. Paragraphs often begin with a precise object or empirical regime, then state what it reveals. The prose is confident about established facts and deliberately modest about extrapolation.

The two main corpus papers differ in polish and age. Prefer the clarity and restraint of `X-regularization/Paper_Arxiv` while retaining the assumption-explicit rigor and quantitative decomposition of `Stochastic-Gradient/Paper`.

## Sentence movement

Use this recurring pattern:

1. State the object or setting.
2. Name the obstacle or distinction.
3. Give the mathematical or algorithmic mechanism.
4. State the quantitative consequence.
5. Interpret the regime where it matters.

Typical useful turns include:

- “The first term …, while the second …” for a genuine decomposition.
- “This isolates …” after a benchmark or derivation separates one mechanism.
- “The conclusion is one-way:” before clarifying what a metric or theorem does not certify.
- “The trade-off is …” when neither method dominates uniformly.
- “Under [assumption], [result].” Keep the condition adjacent to the claim.
- “In our experiments …” for observations not covered by theory.

Do not overuse any fixed phrase. Preserve the logical function, not the wording.

## Paragraph rhythm

Open a section with its task and scope. The first paragraph often names what will be constructed or proved and previews the order of the ingredients. Subsequent paragraphs are medium length and equation-centered: prose sets up a display, the display states the object, and the next prose interprets it.

Use short sentences to mark a sharp limitation or transition. Longer sentences are appropriate for controlled contrasts, but keep the grammatical spine visible.

Avoid generic openings such as “In recent years,” “With the rapid development of,” or “It is well known that.” Avoid rhetorical questions, second person, exclamation marks, slogans, and marketing language.

## Mathematical diction

- Say “Let …” when fixing an object and “Assume …” when imposing a condition.
- Use “satisfies,” “implies,” “yields,” “gives,” and “follows from” with an identifiable mathematical subject.
- Use “characterizes,” “controls,” “bounds,” “decomposes,” and “isolates” only when the formula truly does so.
- Prefer “we construct/prove/show/compare” over vague passive claims.
- Avoid “obviously,” “clearly,” “trivially,” and “easy to see.” Supply the reason.
- Avoid “novel” unless novelty itself has been carefully established. The contribution statement can simply say what is introduced.

## Claims and humility

Match verbs to evidence:

- Theorem or exact derivation: “proves,” “shows,” “equals,” “is bounded by.”
- Controlled numerical evidence: “demonstrates” sparingly; usually “shows” or “matches.”
- Repeated empirical tendency: “tends to,” “is consistently higher on,” “in our experiments.”
- Heuristic: “suggests,” “motivates,” “serves as a proxy.”
- Unproved mathematical belief: “we conjecture,” followed by what remains open.

State negative or mixed results when they delimit the method. The corpus explicitly distinguishes robustness from speed, identifies fake metrics, reports regimes with minor gains, and names failure modes. This candor is part of the style.

## Terminology and notation

Use one term for one concept. Introduce an abbreviation once and use it thereafter. Do not coin a synonym for variety. Match capitalization and hyphenation already established in the paper.

Keep scalars, vectors, measures, maps, and random variables visually distinct according to the local preamble. Reuse existing macros instead of spelling out equivalent LaTeX. Define orientation-sensitive maps and pushforwards explicitly; never assume a reader will infer direction.

## Lists, headings, and emphasis

Connected prose is the default. The corpus uses paragraph headings for compact benchmark-by-benchmark interpretation and uses lists when roles are genuinely parallel. Follow the current section: do not impose `\paragraph`, `itemize`, bold labels, or a contributions list on a paper that does not use them.

Use `\emph` selectively for a named concept or a contrast. Avoid bold prose as a substitute for structure.
