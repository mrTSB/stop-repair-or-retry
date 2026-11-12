# Model-derived lambda policy boundaries

This follow-up checks for policy regimes the 602-point grid could miss. It constructs affine one-call action values at every posterior reachable from all eight initial observation symbols. Their pairwise intersections partition lambda into intervals. Within each interval, each root action value is affine; root intersections refine the partition. The calculation uses model probabilities and costs only, then evaluates the resulting policies on dev only.

Over [0, 1.2], there are 64 enumerated breakpoints and 253 probes (boundaries, interval midpoints, and +/-1e-9 boundary probes). They recover the same eight dev policies as the dense grid. Maximum observed dev accuracy remains 53.33%; the low-token policy with that accuracy begins near 0.24717. No test claims are made. Floating-point arithmetic and the planner's 1e-12 tie tolerance apply.

`protocol.json` records the method, exact boundaries, probes, and hashes before dev evaluation. `dev-regions.json` records all summaries; `dev-policies.json` records each unique path/score/token vector. `validation.json` records the independent construction check. The model and original milestone artifacts are unchanged.
