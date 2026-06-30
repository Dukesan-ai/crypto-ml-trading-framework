# Meta-Labeling Done Right: Orthogonal Information and the Information Advantage

> One of a short series of methodology notes on building leakage-resistant ML
> trading systems. These notes describe *how I think about the problems* — they
> contain no strategy, parameters, or signals.

Meta-labeling is one of the most useful ideas in *Advances in Financial Machine
Learning* (López de Prado, Ch. 3), and also one of the easiest to implement in a
way that quietly does nothing. The mechanics are simple; the design judgment that
makes it *work* is where the value is. This note is about that judgment.

## The architecture, briefly

Meta-labeling splits a trading decision into two models with two different jobs:

- A **primary model** decides *direction* — long or short — and emits a
  conviction along with it.
- A **meta-model** decides *whether to act on that signal at all, and how much to
  size it.* It is a binary problem: given that the primary says "long," is this
  one of the cases where the primary tends to be right?

The meta-model turns a raw directional signal into a sized, risk-managed bet. It
filters out false positives the primary model is prone to, and it converts the
primary's binary call into a continuous position size. Done well, it improves
precision and risk-adjusted return without touching the primary's recall — the
primary keeps finding opportunities; the meta-model decides which ones to take and
how hard.

## The mistake that makes it useless: feeding it the primary's own features

The simplest way to wire up a meta-model — and the one shown in introductory toy
examples — is to hand it the primary model's features plus the primary's
prediction, and let it learn. This almost always underdelivers, and the reason is
worth stating precisely: **there is no information advantage.** If the meta-model
sees the same features as the primary and is the same class of model, it has
nothing new to learn from. It can only re-derive what the primary already knew. At
best it relearns the primary's own decision boundary; it does not add a second,
independent judgment. You have spent the complexity of a two-model system and
received approximately one model's worth of information.

The failure is silent. The pipeline runs, the meta-model trains, the numbers look
plausible — but you have effectively bagged the primary model rather than added a
genuinely separate filter. Recognizing this requires asking not "does the
meta-model train?" but "does the meta-model have access to information the primary
*couldn't use*?"

## The right way: orthogonal information

The meta-model earns its place only if it consumes information the primary model
could not exploit. Concretely, the literature on the framework (e.g. Joubert,
"Meta-Labeling: Theory and Framework," 2022) describes the meta-model's inputs as
roughly four groups, and notably the primary's *raw features* are not the
emphasis:

1. Features predictive of *when the primary tends to be wrong* — for example,
   conditions under which its signal degrades.
2. The primary model's own *evaluation data* — its conviction, and its recent
   reliability/calibration. How sure is it, and how trustworthy has "this sure"
   been lately?
3. **Market-state / regime** information — the context the primary's
   feature set does not encode.
4. The primary's *prediction* itself.

The unifying principle is that the meta-model's value comes from what the primary
model **leaves on the table.** It should look at the problem from an angle the
primary structurally cannot — most usefully, regime and market-state signals that
are orthogonal to the primary's directional features. That orthogonality is the
whole point: it is the source of the second, independent judgment.

## A principle I enforce as a hard constraint: primary features never enter the meta-model

Because the failure mode is silent, I treat "the primary model's own features must
never be inputs to the meta-model" as a hard, self-imposed constraint rather than
a preference. It is easy, in a large feature pool, for a feature the primary
already relies on to leak into the meta-model's inputs, at which point you are
quietly back to the no-information-advantage case. Enforcing strict separation
keeps the two models genuinely independent and keeps the meta-model honest about
where its edge is supposed to come from.

There is an empirical payoff to this discipline that I find genuinely satisfying.
When the meta-model is allowed to select its inputs from a broad pool under this
orthogonality constraint, it consistently gravitates toward signals the primary
model *discards as noise* — features the primary's own importance ranking throws
away. That is the information-advantage thesis made concrete: the meta-model
profits precisely from structure the primary treats as irrelevant. Two models,
two genuinely different views of the same market.

## Long and short are not mirror images

A further design choice: handle long and short with **separate meta-models**
rather than one symmetric model. The informational drivers of up-moves and
down-moves are not mirror images of each other — the dynamics of a rally and the
dynamics of a drawdown differ in structure, in speed, and in which signals carry
information. Forcing one model to serve both directions blends two different
problems and dulls both. A two-model, asymmetric architecture lets each side
specialize. The cost is more models to maintain and thinner data per model; the
benefit is that each meta-model learns the conditions specific to its direction.

## From probability to position size

The meta-model's output is a calibrated probability that the primary is right.
That probability is not just a yes/no gate — it is the natural sizing signal. A
higher, better-calibrated probability maps to a larger position; a marginal one
maps to a small or zero bet. This is why calibration matters so much for the
meta-model and why a proper scoring rule (not raw accuracy) is the right thing to
optimize when building it: the *quality of the probability*, not merely the
side of the call, is what becomes the bet size. The meta-model thus does both of
its jobs — filtering and sizing — through one well-calibrated number.

## Summary

Meta-labeling is not "add a second model"; it is "add a second model that sees
what the first one couldn't." The discipline that separates a meta-model that
works from one that runs and does nothing is the **information advantage**: feed
it orthogonal regime and market-state information plus the primary's conviction
and reliability — never the primary's own features — and enforce that separation
as a hard constraint, because the failure mode is silent. Split long and short so
each learns its own dynamics, and let the meta-model's calibrated probability
become the position size. The value of the architecture lives entirely in the
orthogonality; protect it and the two models give you two genuinely independent
reads on the market.

---

### References
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018 (Ch. 3: labeling and meta-labeling).
- J. Joubert, "Meta-Labeling: Theory and Framework," *Journal of Financial Data Science*, 2022.
