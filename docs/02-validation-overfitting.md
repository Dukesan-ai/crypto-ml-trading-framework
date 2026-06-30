# Separating Real Edge from Overfitting: How I Design Validation

> One of a short series of methodology notes on building leakage-resistant ML
> trading systems. These notes describe *how I think about the problems* — they
> contain no strategy, parameters, or signals.

The single hardest problem in quantitative trading is not finding patterns. It
is telling the difference between a pattern that will persist and one that your
search procedure manufactured out of noise. Everything below is about building a
validation process whose job is to make that distinction as honestly as possible
— and to be hard to fool, including by the person running it.

## Starting premise: in-sample performance is near-zero evidence

A strategy that looks good on the data it was built on tells you almost nothing.
With enough features, enough configurations, and enough flexibility, you can fit
*any* in-sample target. So the in-sample number is not a weak piece of evidence
to be weighted lightly — for practical purposes it is treated as zero. The only
question that counts is whether performance survives out of sample, under
conditions that did not contribute to building the strategy. Every design choice
that follows is downstream of taking that premise literally.

## One backtest is a point estimate; you want a distribution

A single backtest produces one number on one path through history. That number is
fragile: it depends on where the train/test boundary happened to fall, and it
gives you no sense of variance. **Combinatorial Purged Cross-Validation** (CPCV;
López de Prado, AFML Ch. 12) replaces the single path with many. By splitting the
sample into groups and testing every combination of held-out groups, it generates
a *distribution* of out-of-sample outcomes across many paths, each representing a
different slice of history as the test set.

This changes the question you can ask. Instead of "did it work?" — answerable by
luck — you ask "how often, and how consistently, does it work across paths?" A
strategy that is positive on most paths and a strategy that is spectacular on one
path and negative on the rest can share the same average; only the distribution
tells them apart. (See `validation/purged_cv.py` for the splitter.)

## Leakage is the silent killer: purge and embargo

Cross-validation on time series leaks unless you actively prevent it, for two
reasons. First, **overlapping labels**: a label is usually built from a window of
future outcomes, so a training sample whose label window overlaps a test sample's
shares information with it. *Purging* removes those training samples. Second,
**serial correlation across the boundary**: samples immediately after the test
block are correlated with it even without label overlap. An *embargo* drops a thin
band of post-test samples to break that.

The reason to be fanatical about this is that leakage does not announce itself.
It quietly inflates out-of-sample numbers, which is the worst possible failure: it
corrupts the very metric you are trusting to keep you honest. A validation
framework that is not leakage-resistant is not a conservative version of a good
one — it is actively misleading.

## Selection bias: deflate for everything you tried

Suppose you try N configurations and keep the best. Even if none has any real
edge, the maximum of N noisy Sharpe ratios is positive and can look impressive —
that is just the statistics of taking a maximum. The **Deflated Sharpe Ratio**
(Bailey & López de Prado, 2014) corrects for this by deflating the benchmark to
the *expected maximum* Sharpe under N independent trials, then asking whether the
observed Sharpe clears that inflated bar. The **Probabilistic Sharpe Ratio**
underneath it also accounts for short samples and for skew and fat tails, which
otherwise make the Sharpe estimator more uncertain than it appears.

The discipline that matters here is honest accounting: you must deflate by the
number of configurations you *actually searched*, not the one you kept. Under-count
the trials — forget the variants you discarded along the way — and the deflated
metric is itself inflated. The number of degrees of freedom you spent looking at
the data is part of the result. (See `validation/deflated_sharpe.py`.)

## The core idea: the selection metric and the validation metric must differ

This is the part I consider most important, and it is the one most often gotten
wrong. If you select features (or tune a model) to maximize a metric, and then you
validate on that *same* metric, the validation is circular. You have not tested
the strategy against the metric; you have *fit* the metric. Of course it scores
well — you optimized it.

The fix is to make the selection criterion and the validation criterion
deliberately different, and to make the validation criterion the harder one. In
my pipeline, selection is driven by a proper scoring rule — a likelihood-based
criterion that rewards calibrated probabilities and that is not the quantity I
ultimately care about — while validation is a risk-adjusted return gate evaluated
on a partition that played no role in selection. Because the two are different,
the optimizer cannot reach through the selection step and inflate the validation
result; the validation metric was never the thing being maximized. The greater
the conceptual distance between what you optimize and what you trust, the harder
the validation is to game.

A related discipline: hold out a true out-of-sample partition and use it **strictly
as a judge.** Feature selection happens only on the in-sample portion; the holdout
is never consulted to choose anything, only to evaluate the locked result. The
instant you use the holdout to make a decision, it stops being out of sample, and
you are back to fitting your own test.

## The counterintuitive signature: losing the in-sample metric can be good news

Put the previous two ideas together and you get a diagnostic that looks backwards
until you think it through. Compare two candidate feature sets:

- One **wins** on the in-sample selection metric but **loses** out of sample.
- The other **loses** on the in-sample selection metric but **wins** out of
  sample.

The naive reading — "the first one scores higher in-sample, so it's better" — is
exactly wrong. The first set won in-sample precisely because it fit in-sample
noise; that is what overfitting *is*, and the out-of-sample loss is the tell. The
second set could not fit the in-sample noise as well — so it scored lower there —
because what it captured was structure that generalizes, which is why it wins
where it counts. In this regime, **a candidate scoring worse on the in-sample
metric while scoring better out of sample is evidence that it is *not*
overfitting.** If you only looked at the selection metric, you would throw away
the better strategy and keep the worse one. The signature only becomes visible
because the selection metric and the validation metric are different and you are
watching both.

## The meta-principle: every look at the data is a degree of freedom

Underneath all of this is a single idea. Every time you let the data influence a
choice — a feature, a threshold, a model, a configuration — you spend a degree of
freedom, and each degree of freedom is an opportunity to fit noise. Good
validation is the discipline of *minimizing* those opportunities (don't peek at
the holdout; don't re-select on the test) and *accounting* for the ones you
unavoidably spend (deflate by the trial count; treat in-sample numbers as zero
evidence). You cannot eliminate overfitting, but you can build a process that
makes it expensive to fool yourself — and then trust the live, forward record as
the final arbiter, because that is the one set of degrees of freedom you could not
have spent in advance.

## Summary

Treat in-sample performance as zero evidence. Replace the single backtest with a
distribution via purged, combinatorial cross-validation, and be fanatical about
leakage because it corrupts the metric you depend on. Deflate for every
configuration you searched, not just the winner. Above all, keep the selection
metric and the validation metric different so the validation cannot be gamed by
the selection — and once you do, watch for the signature where losing in-sample
but winning out-of-sample is exactly the shape of a strategy that generalizes.

---

### References
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018 (Ch. 7: cross-validation, purging, embargo; Ch. 11–12: backtesting and CPCV).
- D. Bailey & M. López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *Journal of Portfolio Management*, 2014.
