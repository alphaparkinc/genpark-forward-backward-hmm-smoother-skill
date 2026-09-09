from client import ForwardBackwardSmoother

def main():
    print("=== Forward-Backward HMM Smoother ===")
    smoother = ForwardBackwardSmoother()
    states = ('Rainy', 'Sunny')
    obs = ['walk', 'shop', 'clean']
    start_p = {'Rainy': 0.6, 'Sunny': 0.4}
    trans_p = {'Rainy': {'Rainy': 0.7, 'Sunny': 0.3}, 'Sunny': {'Rainy': 0.4, 'Sunny': 0.6}}
    emit_p = {'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5}, 'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1}}

    res = smoother.smooth(obs, states, start_p, trans_p, emit_p)
    print("Smoothed Posteriors:", res)
    assert len(res["smoothed_posteriors"]) == 3
    assert abs(sum(res["smoothed_posteriors"][0].values()) - 1.0) < 1e-3

    print("Forward-Backward Smoother verified successfully!")

if __name__ == "__main__":
    main()
