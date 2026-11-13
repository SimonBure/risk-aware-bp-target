"""Code of "A risk-aware individualized blood pressure target to avoid hypotension in surgery".

- ``model``: the NE PK/PD models (complete, second order, reduced) and their simulation;
- ``wholefit``: the fits of whole records by IPOPT (Section II.A, Fig. 1);
- ``selection``: which parameters of the reduced model to estimate (Section II.A);
- ``fit``: the online fit of the reduced model and its linearised posterior (Section II.B);
- ``forecast``: the law of the next readings, the risk of hypotension and the score;
- ``dose``: the bolus and constant infusion that reach a target (Section II.C);
- ``clinical``: the clinical records and their forecast windows (Section III.A);
- ``virtual``: the virtual cohort, its sensor and its collection (Section III.B);
- ``target``: the personalised target y*, the oracle and the fixed doses (Section III.B);
- ``collection``: the search over collections (discussion);
- ``stats``: the intervals and verdicts of the tables.
"""
