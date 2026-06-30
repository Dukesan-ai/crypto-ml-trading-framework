# When the Right Move Is the Wrong One: Entry Timing, Regime, and Letting the Features Decide

> One of a short series of methodology notes on building leakage-resistant ML
> trading systems. These notes describe *how I think about the problems* — they
> contain no strategy, parameters, or signals.

The previous note ended on an idea I want to take further here: a method can be
individually correct and still produce the wrong decision when applied without
judgment about *when* it should bind. That note treated it as one item on a list.
Running the system through another month of live trading turned it into the thing
I now spend most of my time on. This note is two case studies in it — one where I
built a "better" component and the data told me to delete it, and one where the
obvious architectural move is, I think, a category error. Both come down to the
same discipline: knowing which problems to solve with more model, and which to
solve by *not* adding model at all.

---

## Part I — The optimization that made things worse

### 1. A bug that looked like noise

The trigger was a run of consecutive stop-outs, each at roughly the same loss,
clustered closely in time. The tempting interpretation — the one that lets you do
nothing — is that this is just variance: every classifier has an error rate, and
losing streaks happen. I had written a whole section in the previous note about
the discipline of accepting irreducible error, so I was primed to shrug.

It was not noise. It was a bug, and an instructive one, because it lived entirely
in the gap between how the model was trained and how it was actually being
executed — the train/serve skew the previous note warned about, but in a form I
had not caught.

In research, the label and the barriers were anchored to one specific price: the
close of the last *completed* bar. The model had learned, in effect, "entering at
this reference price, does the position reach its profit barrier before its stop?"
In production, the entry price had drifted somewhere else entirely. A separate
regression model predicted an "optimal" entry offset; that offset was applied to a
*real-time* price pulled mid-bar; and an execution layer could then override the
result again using indicators recomputed off a different bar series. By the time an
order was placed, the actual fill price had almost nothing to do with the
completed-bar close the model had been trained against.

In a rising move the consequences were systematic, not random. Signals tended to
fire as price spiked intrabar; the real-time price sat above the bar's eventual
close; the system entered higher than the backtest assumed; and a stop placed
relative to that higher entry was easier to hit. The losing streak was not the
model being wrong about direction. It was the executed strategy no longer being
the strategy that had been validated. The lesson from the previous note held
exactly: you can do everything right in research and still lose the edge in
delivery, through a door you forgot to close.

### 2. Testing the component instead of trusting the intuition

Here is where it gets interesting, and where I want to be honest about a result
that surprised me.

The fix for the skew is conceptually simple — anchor the live entry to the same
reference the backtest uses, and stop letting a real-time price leak in. But it
raised a sharper question I had been quietly assuming the answer to for a long
time: was the entry-price model *earning its place at all*? It had always felt
obviously useful — surely predicting a better entry point beats just taking the
price in front of you.

I decided to stop assuming and measure it. The entry-price model is a regression
problem (predict an offset), which is a different objective from the classifier
that picks direction — so it deserves its own evaluation, on its own terms, rather
than inheriting trust from the rest of the system. I set up a clean comparison.
The judge was deliberately unflattering to the fancy option: every variant paid
the *worst* execution cost (the full taker fee), so that nothing could win merely
by saving on fees. That isolates the only question that matters — does predicting
an entry price actually find better *points*, independent of cost? Three layers:

- **Baseline.** Take the completed-bar reference price directly. Always fills.
- **Entry model, full features.** Predict an offset, then only fill if the price
  is actually reached on the next bar — otherwise the trade is skipped.
- **Entry model, with its own selected feature subset.** Same, but with features
  chosen specifically for the regression task.

I ran it overnight, five independent times across different windows.

Every single run, the baseline won — by a wide margin, between roughly 12% and 33%
in net terms. The entry-price prediction did not just fail to help; it
*systematically destroyed* value. And the feature-selected version was often
*worse* than the full-feature one, which rules out the comforting explanation that
I had simply picked the wrong features. The paradigm itself — predicting a future
favorable price and waiting for it — had no edge on this data. The model spent its
effort steering toward a worse outcome than doing nothing.

The component I had assumed for months was an asset was a liability. I deleted it.

### 3. Why "just take the price in front of you" wins

The reason is worth stating, because it generalizes. The entry-price model was
trying to predict where price would dip to, and wait there. But waiting introduces
a fork: either price comes back to your level and you fill, or it doesn't and you
miss the trade entirely. The asymmetry of that fork is brutal and one-sided. A
missed trade costs zero — you simply didn't participate. A trade taken at a chased
price costs a real stop-out, *and*, because of the geometry of drawdown recovery,
needs more than one subsequent winner to make back. Skipping a hundred entries is
free; chasing one is expensive. Once I saw it that way, "prefer to miss a trade
than to fill at a price I'm chasing" stopped being timidity and became the correct
expected-value call. The simplest possible execution — take the reference price,
fill or miss — dominates the clever one.

There is a quieter point underneath. I genuinely could not have reasoned my way to
this from first principles; the prior was too strong in the other direction. Only
the measurement, against an honest baseline, overturned it. The discipline isn't
"distrust complexity" as a slogan — it's *building the baseline good enough that a
loved component has to beat it, and then believing the result when it doesn't.*

---

## Part II — The problem I think you cannot solve by detecting it

The harder problem — the one that, for a strategy like this, sits beneath
everything else — is regime change. The feature that predicts well in the market
the model was trained on can quietly invert its relationship to returns when the
market shifts to a state the training data barely contained. This is not a bug you
fix; it is a direct violation of the assumption underneath all supervised learning
— that the distribution you trained on resembles the one you will trade. Most other
problems (a feature, a parameter, an execution path) assume that foundation holds.
Regime change moves the foundation.

The obvious architectural response — and I see it proposed constantly — is to
*detect* the regime first, then switch to a model or feature set fitted for it. A
volatility classifier, a hidden Markov model, a handful of states; route
accordingly. I now believe this is, for the most part, a category error, for three
reasons.

**Detection is defined after the fact, so live detection lags exactly when it
matters.** "Trend," "chop," "crisis" are labels you can draw cleanly only looking
backward. Standing in the present, the regime boundary is blurriest at precisely
the moment you most need it — the turning point, where the data looks like both the
old state and the new one. By the time a detector confirms the switch, the turn has
already happened and you have already taken the losses in the new regime. A
detector is accurate in the calm stretches, where you don't need it, and unreliable
at the transitions, which is the only place it would have paid for itself. That is
structural, not a modeling deficiency to be tuned away.

**A few simple state variables carry far less information than the features
already do.** A volatility index, or a three-state hidden Markov model, compresses
the entire market state into one or two scalars. But a feature set built for this
problem already contains many descriptors of market state — volatility structure,
order-flow descriptors, structural-break statistics, bubble-detection measures.
Asking one or two scalars to do the regime judgment is a *downgrade* from the
high-dimensional, finer-grained picture the features already provide. You'd be
replacing something rich you already have with something coarse you bolted on.

**Most fundamentally: "what regime is this" and "what should I do" are not two
problems — they are one, and a tree ensemble already fuses them.** Every split in a
decision tree is, structurally, exactly this: *if volatility is above X and
order-flow imbalance is beyond Y* — that clause is regime identification — *then
lean this way* — that clause is the conditional response. The branching structure
of a random forest is a regime-recognizer and a conditional-policy welded into one
object. Splitting it into "detect (model A) then act (model B)" doesn't add
capability; it *breaks* the fusion, introduces two error surfaces where there was
one, and discards the interaction between them.

So the conclusion I've reached is to stop trying to predict the regime and instead
make the model *robust* to it — to let the features do the work implicitly,
continuously, inside the model, rather than have a separate, lagging detector do it
explicitly and discretely. Concretely that means three things, none of which is a
detector:

1. **Cover enough regimes in the data.** A forest can only handle implicitly a
   regime it has actually seen. The single highest-value thing isn't a cleverer
   model — it's training data that genuinely spans trending, falling, and choppy
   markets, so the branch structure encodes a real range of conditional responses.
   This is the unglamorous answer, and I think it's the true one. It is also why I
   spend most of my time accumulating and curating data rather than adding model
   machinery.

2. **Protect the conditional features the averaged criterion wants to delete.**
   Some features contribute almost nothing on average — they sit idle in most
   regimes — yet are decisive in the specific state they were built for. A
   selection metric that ranks by *average* importance will discard them; a
   standalone importance number cannot see a contribution that only appears in one
   condition. These features are precisely the raw material the forest uses to
   recognize a regime implicitly. Keeping them out of the averaged cull is keeping
   the model's eyes open. (This connects directly to the conditional-feature point
   in the previous note — the same blind spot in the averaged criterion, viewed
   from the regime side.)

3. **Accept that transitions will hurt, and put the effort into bounding the
   damage rather than predicting the turn.** If detection necessarily lags at the
   turning point, then "not losing at the turn" is unattainable — it is part of the
   irreducible floor. What is controllable is the *size* of the loss when the
   regime shifts: position sizing that scales with conviction, stop discipline that
   bounds the downside. Not predicting the change — surviving it.

There is a constraint worth naming, because it shapes all of this. The raw
information sources a strategy like this can draw on are limited — price, volume,
order flow, and a few others; bars and indicators are all transformations of the
same underlying stream and create no new information. So the leverage is not in
finding a secret input nobody else has. It is in *representation* — how efficiently
the available information is expressed so the structure becomes visible to the
model — and in *selection* — which descriptors stably carry signal across regimes.
A regime detector adds neither. Richer, better-chosen state features do.

---

## What both cases share

The entry-price model and the regime detector look unrelated, but they are the
same mistake wearing two costumes. In both, there is an action that *looks*
obviously correct — predict a better entry, detect the regime before acting — and
in both, the correct move is to *not* do it: take the price in front of you, and
let the features carry the regime implicitly. The judgment is the same one the
previous note circled: a method being valid in isolation tells you nothing about
whether applying it, here, helps. Sometimes the highest-skill decision is to leave
the cleverer component unbuilt — and the only way I've found to know which case
I'm in is to build an honest enough baseline that the clever thing has to beat it,
and then to actually believe the answer.
