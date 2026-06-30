# Feature Selection on Financial Data: Why a Single Run Is Not Deployable

> One of a short series of methodology notes on building leakage-resistant ML
> trading systems. These notes describe *how I think about the problems* — they
> contain no strategy, parameters, or signals.

## The symptom

Run a feature-selection procedure on a financial dataset. Note the subset it
returns. Now re-run it — same data, a fresh random seed, perhaps one or two new
bars appended. You will very often get a *different* subset: some features drop
out, others appear, and the size can swing substantially.

The naive reactions are both wrong. The first is to treat the latest run as the
answer and deploy it. The second is to conclude the procedure is broken. Neither
is right. The instability is real, it is not a bug, and understanding *why* it
happens is what tells you which object is actually safe to deploy.

## The cause: the substitution effect

Financial features are heavily correlated with one another — momentum measured
over nearby windows, volatility estimated different ways, microstructure proxies
that move together. When you measure feature importance under this correlation,
importance gets **split** across correlated groups. This is López de Prado's
*substitution effect* (AFML, Ch. 8): two interchangeable features each receive
roughly half the credit, so each individually looks weaker than it is.

The practical consequence is that, outside a small set of genuinely dominant
features, a large middle band of features sits in a near-tie on importance. The
*rank* within that band is dominated by noise — a feature can land in the top
half on one run and the bottom half on the next purely because a correlated twin
happened to absorb the credit this time. A greedy or recursive selector walking
that band is, in effect, flipping coins among interchangeable candidates. Change
the seed, change the coin flips, change the subset.

So the instability is not the selector failing to find "the" subset. It is the
selector faithfully reporting that, given this data's correlation structure,
**many subsets are roughly equivalent.** The variability *is* the signal.

## The diagnostic that rules out drift

Before blaming correlation, you have to rule out the obvious alternative: that
the subset changed because the *data* changed (regime drift, new samples carrying
new information). The test is simple. Hold the sample fixed — same rows, same
count — and vary only the random seed. If the subset still changes, the cause
cannot be the data; it can only be stochasticity in the procedure interacting
with the correlation structure. In my experience this test comes back
unambiguous: identical data, different seed, materially different subset. That
isolates the mechanism and stops you from chasing phantom "drift" that is really
just the substitution effect.

## The fix: consensus, not any single run

If no single run is trustworthy, the deployable object is what is *consistently*
chosen across many runs. This is **Stability Selection** (Meinshausen &
Bühlmann, 2010, *JRSS-B*): repeat selection over many resamples or seeds, record
how often each feature is chosen, and keep the features that survive across runs.
The crucial property a single run lacks is error control — stability selection
comes with a bound on the expected number of falsely selected features, which a
one-shot procedure simply does not have.

The mental shift is from *"what did this run select?"* to *"what does this data
keep selecting?"* The first is a sample from a noisy distribution. The second is
the location of that distribution — and that is the thing worth deploying.

## The caveat that keeps you honest: vote-splitting

Here is where most write-ups stop, and where the interesting subtlety begins.
The same correlation that destabilized the single run also distorts the
*consensus frequencies.* If a genuinely informative signal is represented by
three correlated features, they split the vote three ways — each ends up with a
lower selection frequency than a lone, uncorrelated feature of equal true
importance. Read frequencies naively and you will under-rate exactly the signals
that have redundant representation. This is the motivation behind **Cluster
Stability Selection** (Faletto & Bien, 2022) and, equivalently, de Prado's
**Clustered Feature Importance** — cluster first, then select at the cluster
level so correlated twins are not competing for the same votes.

So consensus is not a free pass. The frequencies are informative but biased, and
treating "selected 7/10 times" as a clean importance score repeats, one level up,
the same mistake of trusting a noisy number.

## The resolution: meaning comes from validation, not from frequency

The way out of the regress is to stop asking the frequencies to mean more than
they can. Frequencies *nominate*; they do not *confirm.* The consensus core is a
**candidate** — a small, stable set of features the data keeps returning to. You
then take that locked candidate and measure its performance directly, out of
sample, the same way you would judge any deployable artifact (see the companion
note on validation, and `validation/`). The frequency told you *what to test*;
the out-of-sample result tells you *whether it works.* The number that carries
weight is the validated performance of the locked set — not the vote count that
proposed it.

This also dissolves the vote-splitting worry for deployment purposes: you are no
longer interpreting frequencies as importances, only using them to propose a set,
which is then validated on its own merits.

## The principle underneath: stability is a property, not an objective

It is tempting, having seen instability cause trouble, to start optimizing *for*
stability. That is a trap. A feature set can be perfectly stable and completely
useless — fix any arbitrary set and it is, trivially, 100% stable across runs.
Stability is only meaningful jointly with predictive performance; on its own it
is not a goal. The correct role for stability is as a property a *deployable*
object should have, and as a tie-breaker among options that are otherwise
statistically indistinguishable on performance — prefer the more stable, simpler
one. Selection is driven by out-of-sample performance; stability decides ties.
Reverse that ordering — make stability the objective — and you will happily
deploy something stable and dead.

## Summary

On thin, correlated financial data, single-run feature selection is a coin-flip
among interchangeable features, and the variability is the data's correlation
structure talking, not a defect. Rule out drift with a fixed-sample seed test;
recover a deployable object with cross-run consensus; stay honest about
vote-splitting by remembering that consensus frequencies are biased; and resolve
the whole thing by letting out-of-sample *validation* — not selection frequency —
be what confers meaning. Stability is a property of a good answer, never the
objective itself.

---

### References
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018 (Ch. 8: feature importance; substitution effect; Clustered Feature Importance).
- N. Meinshausen & P. Bühlmann, "Stability Selection," *Journal of the Royal Statistical Society: Series B*, 2010.
- G. Faletto & J. Bien, "Cluster Stability Selection," 2022.
