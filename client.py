class ForwardBackwardSmoother:
    """Forward-Backward algorithm for HMM state smoothing."""
    def smooth(self, obs: list[str], states: tuple[str, ...], start_p: dict, trans_p: dict, emit_p: dict) -> dict:
        n = len(obs)
        if n == 0:
            return {"posteriors": []}

        # Forward pass
        fwd = []
        f0 = {s: start_p[s] * emit_p[s].get(obs[0], 1e-6) for s in states}
        fwd.append(f0)

        for t in range(1, n):
            ft = {}
            for s in states:
                ft[s] = sum(fwd[t-1][prev] * trans_p[prev][s] * emit_p[s].get(obs[t], 1e-6) for prev in states)
            fwd.append(ft)

        # Backward pass
        bwd = [None] * n
        bwd[n - 1] = {s: 1.0 for s in states}
        for t in range(n - 2, -1, -1):
            bt = {}
            for s in states:
                bt[s] = sum(trans_p[s][nxt] * emit_p[nxt].get(obs[t+1], 1e-6) * bwd[t+1][nxt] for nxt in states)
            bwd[t] = bt

        # Smoothed posteriors
        posteriors = []
        for t in range(n):
            norm = sum(fwd[t][s] * bwd[t][s] for s in states)
            post_t = {s: round((fwd[t][s] * bwd[t][s]) / norm, 4) for s in states}
            posteriors.append(post_t)

        return {
            "sequence_length": n,
            "smoothed_posteriors": posteriors
        }
