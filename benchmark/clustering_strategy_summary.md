# Clustering Strategy Summary

This report is the final landing page for the RedCEA clustering proposal benchmark.

## Questions

- Which method is best for VDJdb?
- Which method is best for YFV?
- Does CDR3 length explain density heterogeneity?
- Do known YFV-associated clonotypes fall into significant enriched clusters?
- Does enriched output increase cross-donor sharing compared with raw day-15 repertoires?
- Is there one robust default proposal backend, or should RedCEA keep dataset-specific clustering backends?

## Interpretation Template

Populate this report after running notebooks `04` through `08`.

- If Leiden wins on curated VDJdb while VDBSCAN wins on YFV, conclude dataset-specific backends remain preferable.
- If one of `dbscan`, `vdbscan`, `vdbscan_leiden`, `leiden`, `leiden_dbscan`, or `hierarchical_leiden` is strongest on both datasets, promote it as the default.
- If hybrids do not improve stability, keep the proposal step modular and dataset-dependent.
