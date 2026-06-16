# Attribution

This repository studies boundary prediction for IFEval-style verifiable
instruction-following tasks.

IFEval was introduced by Zhou et al. as an objective and reproducible
benchmark for instruction following, based on automatically checkable
constraints such as length, keyword, format, punctuation, and similar rules:

- Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., & Hou, L. (2023). Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911.
- Google Research IFEval implementation and data: `google-research/google-research/instruction_following_eval`.

This repository does not report a standard IFEval leaderboard score. It uses
the IFEval-style checker setting to define a different supervised problem:
given only a prompt `x`, predict whether a fixed local target model `T` will
produce an answer that passes a deterministic checker.

Where applicable, prompt formats, constraint families, and checker logic are
derived from or inspired by original IFEval code and data. Modifications in
this experiment include target-model relabeling with Qwen3-4B-Instruct-2507,
split construction for boundary prediction, and supervised training of
prompt-only boundary models.

The original IFEval data is released under CC BY 4.0, and the source code is
released under Apache 2.0 by Google Research. Vendored checker files in
`boundary-if/src/open_instruct/IFEvalG/` retain their Apache-2.0 headers.
