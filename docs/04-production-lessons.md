# The eleventh reason ML strategies fail

In *The 10 Reasons Most Machine Learning Funds Fail*, Marcos López de Prado catalogs how
quantitative strategies die: backtest overfitting, misusing the backtest as a research
tool, non-i.i.d. samples, ignoring multiple-testing when evaluating Sharpe, and more.
That list is the hard part. It names the failure modes that sink most strategies before
they ever deserve to trade, and the methods that go with it — purged cross-validation,
the deflated Sharpe ratio, meta-labeling — are the foundation this entire project is
built on. Getting research right is the prerequisite, and it is genuinely difficult; the
list earns its name.

What it deliberately scopes out — because that is a different problem, not because it
was overlooked — is what happens *after* the research is right. This note is a
practitioner's field log of that part: the walls I ran into taking a strategy that
survives honest out-of-sample scrutiny and actually running it live with real money.
Calling it an "eleventh reason" is a bit of a conceit; it is really a footnote to a
better paper. But it has a shape worth naming, and two faces. First, a strategy that is
correct in research can still fail to be **delivered** correctly to the market — the
edge is real, but execution, infrastructure, and train/serve consistency quietly eat it.
Second, a correct method is not self-applying: used mechanically, without judgment about
when and how it should bind, **the right method can still produce the wrong decision.**
None of what follows touches signal logic, features, or parameters; it is about
everything that surrounds them — the gap, for one small live system, between a correct
strategy and a profitable one.

---

## Part I — Correct in research, lost in delivery

### 1. The backtest hides an entire class of bug

The most expensive failure I hit was not a bad signal. It was a protective stop that,
in a fast move, failed to fire at all — a position that should have closed at a bounded
loss ran to several times that loss before a separate reconciliation path caught it.

The cause was an order-management error: on each update the stop was being re-anchored
to the *current* price rather than held where it was placed, so as price ran through
the level, the stop kept moving ahead of it instead of triggering. The fix is
elementary once seen — a protective stop must only ever ratchet in the protective
direction, and a breach must close immediately rather than recompute. That part is just
engineering discipline.

The deeper point is *why this was invisible until it cost real money.* A backtest fills
the stop at the barrier price, by assumption. It cannot represent a stop that fails to
trigger, because in the model the trigger is an event, not an order competing with a
moving market. So the entire class of failure — *the order logic that turns a decision
into a fill* — is structurally absent from the thing we use to validate strategies.

This is what makes it dangerous, and why it belongs on the list rather than in a bug
tracker. "Validated out-of-sample" feels like total coverage, but it only ever covers
the **decision** layer — does the model call direction correctly. It says nothing about
the **execution** layer — does the order behave correctly when it meets a real book in
real time. Most discussions of backtest limitations focus on *overstated returns*. The
subtler hazard is that the backtest renders execution ideal, and in doing so makes an
entire family of bugs unobservable in research. Real losses frequently live in exactly
the layer the backtest cannot see.

### 2. Train/serve skew — research contamination, re-entering through the back door

If the first failure is the one the backtest can't see, this is the one *nothing* can
see — no stack trace, no error log, every dashboard green. The features computed at
inference time drift away from the features the model was trained on, and the model
quietly makes worse decisions than it should.

It hides in small places. A rolling statistic computed slightly differently online than
in the research batch. A cached quantity that refreshes on one side but not the other. A
lookback window one bar shorter live than in training. A fallback value that differs
between the two paths. None of these raise an error. They simply present the model, in
production, with inputs drawn from a subtly different distribution than the one it
learned — and a model is only valid on the distribution it was trained on.

What makes this the sharpest item on the list is its relationship to de Prado's own
warnings. His list is about contamination *in research* — leakage and non-i.i.d.
structure corrupting the model before it is ever trusted. Skew is the same corruption
**reintroduced in production, through the back door, after you have already validated
the model correctly.** You can do everything on his list right — purge, embargo, deflate,
hold out an honest test set — and still lose the edge here, because the model you so
carefully validated is no longer the model that is running. The clean object you tested
and the object now trading have silently diverged.

The discipline that contains it is to treat the feature-computation path as a
**contract** between training and serving: anything that changes how a feature is
produced — a cache, an incremental update, a window, a fallback — must be provably
identical on both sides, ideally verified by reconciling live-computed features against
a batch recomputation to a tight numerical tolerance. A crash you can fix in an hour.
Skew can bleed the edge for weeks before you suspect it exists. Of all the failures
here, this is the one large, well-resourced teams build entire systems (feature stores,
parity tests) specifically to prevent — which is the clearest sign that it is hard, not
incidental.

---

## Part II — Correct method, wrong decision

The second face of the eleventh reason is subtler than any bug. The methods de Prado
prescribes are correct. But a method is a tool, not a verdict, and applied
mechanically — without judgment about *when* and *how* it should bind — the right method
yields the wrong decision. These are the mistakes I had to learn not to make.

### 3. Knowing when *not* to let a rule override the model

The instinct, when a model does something you dislike, is to bolt on a hard rule: a
cooldown after two consecutive losses; a block on entering after a sharp run-up. Both
feel like prudent risk management. Both are usually a mistake — and recognizing why is a
genuine test of understanding.

A cooldown after consecutive losses assumes the losses carry information about the next
trade. In a system with positive expectancy, a run of losses is the *expected* behavior
of a random sequence — variance, not a signal. Halting after two is the gambler's
fallacy in code: it overrides a model that was right to keep trading, on the basis of
noise. The "chasing a breakout" problem is the same error in a different costume —
the urge to forbid late entries with a rule. But the model entered late because, at that
moment, the signal said enter; the failures that follow are the cost of a momentum style,
not a malfunction a rule can patch.

The discipline is to resist patching the model's *outputs* with hand rules that encode
your own discomfort. If a behavior is genuinely wrong, the fix belongs upstream — in the
features the model sees, or in how the label is defined — not in a downstream override
that substitutes your bias for the model's calibrated judgment. A hard rule should bind
only where it encodes a real constraint (a risk limit, a hard cost), never where it is
just an emotional reaction to ordinary variance. Knowing the difference is most of the
skill.

### 4. The averaged criterion that throws away conditional features

Feature selection ranks features by their *average* contribution. That sounds neutral,
but it encodes an assumption — that a useful feature is useful most of the time. In
markets, the most valuable features are often the opposite: dormant in the common case,
decisive in the rare one. An over-extension measure contributes almost nothing while
price behaves, then everything when price has run too far. Averaged over the whole
sample, its score is unremarkable, and a selector that ranks by mean importance prunes
it — precisely because it is conditional rather than constant.

The lesson is that the selection *criterion* carries a hidden model of what "important"
means, and the default — average contribution — is systematically blind to
conditionally-useful features. The features it discards are not noise; they are the ones
that earn their place only in the regimes that matter most, which is exactly when you
most need them. Recognizing this changes what you do: the answer is not to lower the
selection bar (that readmits real noise), but to protect features with a clear,
*a priori* reason to be conditionally decisive, and to give the model the regime context
(state features) it needs to use them only when they apply. This is also why a single
selection run is not the deployable object — it has to be cross-validated for genuine,
out-of-sample conditional value, or you are merely overfitting a different way.

### 5. Reading your own validation numbers correctly

A subtle trap is misreading the output of your own validation machinery. Combinatorial
purged cross-validation produces, by construction, far more total PnL than a single
held-out test — and it is easy to read that larger number as the stronger result. It is
not; it is an artifact of *how* CPCV works.

CPCV does not run the data once. It builds many train/test splits across combinations of
groups, so each observation is tested many times, across many paths. The aggregate PnL
is the sum over all of that — many folds, many paths — so of course it dwarfs the PnL of
a holdout that tests a small slice exactly once. The two numbers are not on the same
scale and were never meant to be compared in absolute terms. What you compare are
scale-free quantities — Sharpe, return per unit risk, hit rate — and even then with the
holdout treated as the more honest judge, because the holdout's data was isolated from
selection entirely while CPCV's was not.

The general lesson is that a validation method has a *grammar*, and reading its output
without that grammar produces false confidence — here, mistaking a sample-count-times-
test-count artifact for evidence of edge. Knowing how the tool produces its number is
inseparable from knowing what the number means; the number alone, taken at face value,
misleads.

### 6. The selection metric must not be the validation metric

A foundational discipline, and the one most easily violated by accident: the metric you
*select* features on must differ from the metric you *trust* to validate. If you choose
features to maximize a quantity and then judge the result by that same quantity, the
judgment is compromised — you optimized the thing you are now using as evidence, and the
result reflects the optimization, not genuine edge.

The fix is to keep a conceptual distance between the two. Select on one criterion (a
proper scoring rule on the calibration of the model, say), and validate on another
(risk-adjusted return on a held-out path the selection never touched). The greater the
distance between *what you optimize* and *what you trust*, the harder it is for the
optimization to reach through and inflate the verdict — and the more an out-of-sample
win means something rather than echoing the in-sample fit. This is what makes the
counter-intuitive signature trustworthy: a subset that scores *worse* on the selection
metric in-sample but *better* on the validation metric out-of-sample is showing you
generalization, not fit. If selection and validation were the same metric, you could
never see that shape, because the two could never disagree.

### 7. The discipline of accepting irreducible error

The last item underwrites all the others. Every classifier has a floor — an irreducible
error set by how much the classes overlap in feature space. Two situations can present
identical features and resolve to opposite outcomes; no model, however good, separates
them, because the information that would separate them is not in the inputs. A breakout
that holds and a breakout that fails can look the same at the moment of entry. Some
fraction of losses is therefore not a defect to be engineered away — it is the cost of
operating in a domain where the future is genuinely not a function of the present.

This reframes what "improving the model" even means. The error you face splits into two
parts: the reducible part, which better features or better labels can lower, and the
irreducible part, which they cannot. Confusing the two is itself a path to failure —
chasing the irreducible part means fitting noise (overfitting in the name of
improvement), while ignoring genuinely reducible error means leaving edge on the table.
The skill is to tell them apart: an error scattered across every condition, where each
bucket already breaks even, is the floor — accept it. An error clustered in an
identifiable situation is reducible — fix it, upstream, with information. And it closes
the loop with reason #3: the urge to bolt a rule onto every loss is, at bottom, a
refusal to accept the irreducible floor — trying to patch with hand-rules what is not a
flaw but the nature of the problem.

---

## Why this note exists

De Prado's ten reasons get a strategy to the point of being correct in research, which
is the hard, foundational part and the prerequisite for everything else. This note is
about the stretch that comes after, for one small system run by one person: order logic
that has to survive a real order book, train/serve consistency that keeps the live model
seeing what it was trained on, and the judgment to apply correct methods correctly — to
tell when a rule helps from when it is the gambler's fallacy in disguise, which features
earn their place only in the regime that matters, how to read your own validation
numbers, and which errors are worth fixing versus the cost of doing business.

Backtest numbers are easy to show and easy to fool yourself with — which is exactly why
de Prado's list exists, and why none of the lessons here are about chasing a prettier
equity curve. They are the unglamorous half: the part where a validated edge either
makes it intact to the market or doesn't, and where good methods are either applied with
judgment or misapplied with conviction. I wrote it down because it is where most of my
actual time has gone, and because I had to learn most of it by losing money I didn't
have to lose. If it saves someone else a few of those tuitions, it has done its job.
