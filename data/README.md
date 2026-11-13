# Clinical data

The clinical records of Joachim et al. (2024) are identifying health data under the GDPR and
French law (loi Jardé); they cannot be shared openly. They are available from the corresponding
author upon reasonable request, subject to approval by AP-HP.

With the data, place `data.csv` (MAP and plasma concentrations) and `injections.csv` (NE boluses
and infusions) here, or point the environment variable `BPTARGET_DATA` to their folder; the format
is described in `bptarget/clinical.py`. Only the drivers `drivers/clinical_*.py` need them.
