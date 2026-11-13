"""Fig. 2: one decision at t = 10 min with alpha = 5 %, on two virtual patients of the paper's draw.

Each column is one patient. Top: the collection's readings and the fit up to t; from t, the forecast
of the readings under the dose evaluated (mean and pointwise 90 % band), the rise T_r and the window
W_t, the 65 mmHg threshold. Bottom: the NE rate, the collection then the dose (its bolus over 10 s drawn
clipped). Left, patient 5, given NE: y* and its dose applied to the true patient. Right, patient 109,
refused: the forecast under the highest dose the fit allows, whose band dips below 65 mmHg; dotted, the
true MAP had the dose been given.

Run ``python drivers/decision.py [--patients 5 109]``: writes ``outputs/decision.{pdf,png}`` and prints
the numbers shown.
"""
import common  # noqa: F401, first: one thread per worker

import argparse

import matplotlib.pyplot as plt
import numpy as np

from bptarget.dose import DEADLINE, HORIZON
from bptarget.fit import N_BOX, PatientModel, fits
from bptarget.forecast import HYPO, risk, smallest_target
from bptarget.target import setup
from bptarget.virtual import SEED, collection, draw, noise, record, sample_times, shown, true_map, with_dose

T = 600.0          # s, the decision time
ALPHA = 0.05
RATE_CLIP = 0.8    # nmol/s, the NE axis: the bolus is drawn clipped


def case(i: int, params: dict) -> dict:
    """Virtual patient i at t: the fit, y* (None when refused), the dose evaluated (y*'s, or F(u_max)'s
    when refused), its forecast over the record, and the true MAP under it."""
    patient = record(i, params)
    model = PatientModel(patient, N_BOX)
    fit = next(f for f in fits(patient, model, until=T) if abs(f[0] - T) < 1e-6)
    eta, sigma2, p = fit[2], fit[3], fit[4]
    at, top, dr = setup(i, params, fit, model)
    y_star = smallest_target(lambda y: risk(at(y)[1], HYPO, dr), ALPHA, HYPO, top)
    y = top if y_star is None else y_star
    ctrl, pred = at(y)
    u = with_dose(collection(T), T, ctrl.bolus, ctrl.rate)
    times = sample_times(T + HORIZON)
    mean, s = PatientModel(shown(i, u, times, np.zeros(len(times))), N_BOX)(eta)
    grid = np.arange(0.0, T + HORIZON + 0.5, 1.0)
    obs = patient.map
    return dict(i=i, refused=y_star is None, y=y, top=top, risk=risk(pred, HYPO, dr), bolus=ctrl.bolus,
                rate=ctrl.rate, sigma2=sigma2, u=u, times=times, mean=mean,
                sd=np.sqrt(np.einsum("ij,jk,ik->i", s, p, s) + sigma2), grid=grid, true=true_map(params, u, grid),
                readings=true_map(params, u, times) + noise(SEED, i, 0, T + HORIZON),
                fit_t=obs.times[obs.times <= T], fit_y=obs.values[obs.times <= T])


def panel(top, low, c: dict, letter: str) -> None:
    m, mg = c["times"] / 60, c["grid"] / 60
    t, rise, end = T / 60, (T + DEADLINE) / 60, (T + HORIZON) / 60
    after = m > t
    top.axvspan(t, rise, color="white", hatch="////", ec="#d9d8d3", lw=0, zorder=0)
    top.axvspan(rise, end, color=common.PALE, lw=0, zorder=0)
    top.axvline(t, color=common.MUTED, lw=0.6)
    top.plot(c["fit_t"] / 60, c["fit_y"], ".", color=common.GRAY, ms=2.2, label="readings")
    top.plot(m[~after], c["mean"][~after], color=common.BLUE, lw=1.1, label="fit")
    band = 1.645 * c["sd"][after]
    top.fill_between(m[after], c["mean"][after] - band, c["mean"][after] + band, color=common.BLUE, alpha=0.16,
                     lw=0, label="forecast, 90 % band")
    top.plot(m[after], c["mean"][after], color=common.BLUE, lw=1.1, ls="--")
    if c["refused"]:  # after t, the true MAP had the dose been given: dotted
        top.plot(mg[mg <= t], c["true"][mg <= t], color=common.INK, lw=0.7, label="true MAP")
        top.plot(mg[mg > t], c["true"][mg > t], color=common.INK, lw=0.7, ls=":")
    else:
        top.plot(m[after], c["readings"][after], ".", color=common.GRAY, ms=2.2)
        top.plot(mg, c["true"], color=common.INK, lw=0.7, label="true MAP")
    top.axhline(HYPO, color=common.RED, lw=0.8, ls="--")
    top.text(end, HYPO, " 65", color=common.RED, fontsize=6.5, va="center")
    name = "$\\hat F(u_{\\max})$" if c["refused"] else "$y^\\star$"
    top.hlines(c["y"], rise, end, color=common.BLUE, lw=0.9, ls=":")
    top.text(end, c["y"], f" {name}\n {c['y']:.1f}", color=common.INK, fontsize=6.5, linespacing=0.95,
             va="top" if c["refused"] else "center")
    on = top.get_xaxis_transform()
    for x, ha, label in [(0.2, "left", "collect, fit"), (t - 0.15, "right", "decide"),
                         ((t + rise) / 2, "center", "$T_r$"), ((rise + end) / 2, "center", "$W_t$: risk")]:
        top.text(x, 0.97, label, transform=on, ha=ha, va="top", fontsize=6.5, color=common.MUTED)
    if c["refused"]:
        msg = (f"highest dose:\n$R_t$ = {100 * c['risk']:.0f} % > $\\alpha$ = 5 %\n"
               "no NE given, the decision\nreturns to the clinician")
        top.text(0.03, 0.80, msg, transform=top.transAxes, fontsize=6.5, color=common.INK, va="top", linespacing=1.1)
        w = c["true"][mg >= rise]
        top.text((rise + end) / 2, 0.86, f"dotted: true MAP had NE been\ngiven, {w.min():.0f}–{w.max():.0f} mmHg (off scale)",
                 transform=on, ha="center", va="top", fontsize=6.5, color=common.INK, linespacing=1.1)
    else:
        msg = f"$R_t(y^\\star)$ = {100 * c['risk']:.1f} % ≤ $\\alpha$ = 5 %: NE given"
        top.text(0.20, 0.05, msg, transform=top.transAxes, fontsize=6.5, color=common.INK, va="bottom")
    top.set_ylabel("MAP (mmHg)")
    top.set_title(f"({letter}) patient {c['i']}, " + ("refused" if c["refused"] else "NE given"),
                  fontsize=7.5, loc="left", color=common.INK)
    r = np.minimum([c["u"](s) for s in c["grid"]], RATE_CLIP)
    before = mg <= t
    low.fill_between(mg[before], 0, r[before], step="post", color=common.GRAY, alpha=0.45, lw=0)
    low.plot(mg[before], r[before], color=common.MUTED, lw=0.8, drawstyle="steps-post")
    low.plot(mg[~before], r[~before], color=common.BLUE, lw=1.0, ls="--" if c["refused"] else "-",
             drawstyle="steps-post")
    label = f"rate {c['rate']:.2f} nmol/s" + (f",\nbolus {c['bolus']:.0f} nmol" if c["bolus"] > 0 else "")
    if c["refused"]:
        low.text(t + 0.25, 0.04, label + ", not given", fontsize=6.5, color=common.INK, va="bottom", linespacing=1.0)
    else:
        low.text(t + 0.25, RATE_CLIP * 0.95, label, fontsize=6.5, color=common.INK, va="top", linespacing=1.0)
    low.set(ylabel="NE (nmol/s)", xlabel="time (min)", ylim=(0, RATE_CLIP * 1.05), xlim=(0, end))
    low.set_xticks(np.arange(0, end + 0.1, 5))


def figure(cases: list[dict], path) -> None:
    common.style()
    fig, axes = plt.subplots(2, 2, figsize=(common.PAGE, 2.9), sharex="col", gridspec_kw=dict(height_ratios=[3, 1.1]))
    for k, c in enumerate(cases):
        panel(axes[0, k], axes[1, k], c, "ab"[k])
    lo = min(min(c["fit_y"].min(), (c["mean"] - 1.645 * c["sd"])[c["times"] > T].min()) for c in cases)
    hi = max(max(c["fit_y"].max(), c["y"], (c["mean"] + 1.645 * c["sd"])[c["times"] > T].max()) for c in cases)
    for k in range(2):
        axes[0, k].set_ylim(min(lo, HYPO) - 2, hi + 6)
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, nargs=2, default=[5, 109], metavar=("GIVEN", "REFUSED"))
    args = parser.parse_args()
    common.OUT.mkdir(exist_ok=True)
    patients, _ = draw(max(args.patients) + 1, [T])
    cases = [case(i, patients[i]) for i in args.patients]
    for c in cases:
        print(f"patient {c['i']}: {'refused' if c['refused'] else 'given'}, target {c['y']:.2f} mmHg, F(u_max) "
              f"{c['top']:.2f}, risk {100 * c['risk']:.1f} %, bolus {c['bolus']:.1f} nmol, rate {c['rate']:.3f} nmol/s, "
              f"sigma_t^2 {c['sigma2']:.2f}")
    figure(cases, common.OUT / "decision")
    print(common.OUT / "decision.pdf")


if __name__ == "__main__":
    main()
