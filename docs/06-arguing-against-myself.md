# Arguing Against Myself: Steelmanning the Schools That Disagree with My Methodology

> One of a short series of methodology notes on building leakage-resistant ML
> trading systems. These notes describe *how I think about the problems* — they
> contain no strategy, parameters, or signals.

The previous notes describe a methodology built largely on López de Prado's
*Advances in Financial Machine Learning* — sparse feature selection, combinatorial
purged cross-validation, deflated and probabilistic Sharpe ratios, meta-labeling.
Anyone can adopt a framework and defend it. The more useful question, and the one
that actually protects you, is the opposite: **what are the strongest arguments
that this framework is wrong for my problem, and do my own results agree with
them?**

So at one point I stopped reading in the direction of my own conclusions and built
a deliberate opposition file: the schools of thought, on the same serious footing
as de Prado, that contradict pieces of what I do. The goal was never to decide "who
is right" in the abstract. It was to find the experiments that could settle each
disagreement *on my own data* — a thin, low signal-to-noise, highly non-stationary
crypto direction-classification task — and to be willing to lose the argument. This
note is about that exercise, and about one disagreement where my own data told me,
for a while, that the other side was right.

## The opposition, taken seriously

Four positions, each of which contradicts or sharpens something I do. I'll state
each as its proponents would, not as a strawman.

**The complexity school** (Kelly, Malamud, Zhou — *The Virtue of Complexity in
Return Prediction*). The conventional wisdom — and de Prado's instinct, and mine —
is that on noisy financial data you should *reduce* dimensionality: prune to a
sparse feature set so the selection itself controls variance. The complexity school
argues, with both theory and broad empirical support, the reverse: in return
prediction, models where the parameter count vastly *exceeds* the number of
observations keep improving out-of-sample as you make them more complex, provided
you control variance with ridge shrinkage rather than with selection. This is a
direct hit on my pruning. They control variance by *shrinking*; I control it by
*choosing* — and de Prado's own warning that "selection search overfits" does not
touch ridge, because ridge makes no discrete choices and has no garden of forking
paths to get lost in. If they are right on my data, the entire feature-selection
spine of my system is the wrong tool, and I should be shrinking, not selecting.

**The multiple-testing school** (Harvey & Liu; White's Reality Check; Hansen's
SPA; Romano–Wolf). This one is an ally, but with sharper tools than I was using. It
agrees with the deflated Sharpe in spirit — discount performance for how many
things you tried — but offers *non-parametric bootstrap* tests that fit a
solo-researcher's "I searched a lot of configurations and features" situation more
directly than a parametric DSR, and without a normality assumption. It also carries
Harvey's most useful hook: a relationship with a genuine economic rationale earns a
*lower* statistical bar, because theory absorbs much of the multiple-testing
penalty. The implication for a weak directional signal is uncomfortable and
correct: the highest-leverage move is not to try a hundred more features, it is to
give the signal a reason to exist.

**The "overfitting is a mathematical certainty" school** (Bailey, Borwein — the
minimum backtest length). Their point is sobering: on *pure noise*, if you try
enough configurations, you are *guaranteed* to find an arbitrarily high backtested
Sharpe. Overfitting past a certain number of trials is not a risk to be managed; it
is a deterministic outcome. This hands me a prior gate I didn't have — before
searching, estimate whether my sample is even long enough that the best result
*could* be real given how much I'm searching.

**The adaptive / decay school** (Lo's adaptive markets; McLean & Pontiff). Edge is
not a static object that holds across folds; it is regime-conditional and decays —
sometimes precisely because it has been found and crowded out. This is the one that
explains a symptom I kept seeing: a holdout result that swings from clearly positive
to clearly negative across snapshots. If directional edge is regime-conditional,
then a single-snapshot holdout *must* be unstable, and chasing a globally stable
signal is chasing something that doesn't exist.

## The disagreement where the other side was winning

The sharpest of these is the complexity school, because it collides head-on with
the most central choice in my system, and because — for a real stretch — my own
data sided with it.

My first honest test was unflattering to me. Across three coherent snapshots, the
*full* high-dimensional feature model held up better out-of-sample than my pruned,
selected subset. Worse, the subset showed the textbook signature of selection
overfitting: it looked far better than the full model *in cross-validation*, and
that entire advantage evaporated on the holdout. The number that vanished between
CV and holdout was almost exactly the size of the subset's apparent edge. By every
honest reading, on my data, at that time, "selection" was adding variance and not
much else. I wrote down the conclusion the evidence supported: *on a weak signal,
maybe I should drop the selection and go back to the full feature set.* The
opposition was beating me on my own turf.

This is the moment that the exercise is actually for. There were two lazy options,
and both are common. One is to wave the result away — *my framework says pruning is
right, so this must be noise.* The other is to capitulate immediately — *the data
says full features win, so flip the whole system to shrinkage.* I think both are
failures of the same kind: treating "the other side is ahead" as a verdict to be
accepted or denied, rather than as a phenomenon to be *diagnosed*.

So I asked the narrower question: *why* was selection losing? And the answer was not
"pruning is wrong in principle." It was that my pruning *method* was bad. The greedy,
single-pass selection I was using was itself a source of the overfitting — it was
chasing noise in the ranking, producing a different subset every run, and the
instability underneath wasn't really resampling noise, it was *regime* instability
that my procedure had no defense against. The complexity school had correctly
identified a real failure; it had just mislabeled the cause. The failure was in *how
I was selecting*, not in the decision to select at all.

That reframing pointed at a fix rather than a surrender. I rebuilt the selection
around cross-run stability — keeping only what survives across many subsamples
(stability selection), clustering correlated features so the substitution effect
couldn't split their votes, and adding a correction aimed specifically at the
regime instability that was the true culprit. With a selection procedure that was
actually robust, the result reversed: the subset now beat the full feature set
across every one of thirteen rounds. The earlier conclusion was overturned — not by
ignoring it, but by taking it seriously enough to find what it was really pointing
at.

And then one more turn, because the honest version of this story doesn't end on a
clean win. Digging into *why* the full-feature model had looked competitive earlier,
I found that under the full feature set the second-stage model was quietly
degenerating — collapsing to a near-constant output, effectively failing to train.
Part of "full features wins" had been an artifact of a broken downstream model, not
a real virtue of complexity. The cleaner reading was almost the opposite of where I
started: reducing dimensionality was what let the rest of the system work at all.

## Where it actually landed

The place I ended up is not "selection beats complexity" — that would just be
re-planting my flag. It is something more specific, and I think more honest. On thin,
non-stationary data, the subset that "wins" changes from run to run because many
features are mutually substitutable; the meaning therefore cannot live in which
features any single run picks. It has to live in what survives across runs, and in
what holds up on an out-of-sample judge. The resolution wasn't choosing a side in
"more features versus fewer." It was realizing the question was posed at the wrong
level, and moving to cross-run consensus and validation as the thing that actually
carries the signal.

I want to be careful about the scope of even this. The complexity school's evidence
is real; it is largely drawn from *return regression* — predicting aggregate returns,
forming portfolio weights — not from single-trade directional classification with
triple-barrier labels, and its own later work concedes that the complexity dividend
is bounded by non-stationarity, which a crypto instrument has in abundance. That my
data favored selection is a statement about *my task*, not a refutation of their
result on theirs. Both can be true. That is usually the actual shape of these
disagreements, once you stop trying to win them.

What I took from the whole exercise is less a conclusion than a stance. A framework
you only ever argue *for* is a belief. A framework you have genuinely tried to break
— by building the opposition's strongest case, running the experiment that could
have proven you wrong, and being willing to follow the result — is something you can
actually stand behind, including in the places where you choose to keep it. The
value wasn't in confirming what I do. It was in knowing, for each piece, exactly
which experiment would change my mind, and having already run it.
