def print_metrics(results):
    """Pretty print tracking metrics results dynamically based on available metrics."""
    print("\n" + "=" * 60)
    print("TRACKING METRICS RESULTS")
    print("=" * 60)

    # Group metrics by prefix
    grouped = {}
    nested = {}

    for key, value in results.items():
        if isinstance(value, dict):
            nested[key] = value
        else:
            # Extract prefix (e.g., 'CTC' from 'CTC_TRA')
            if "_" in key:
                prefix = key.split("_")[0]
                suffix = "_".join(key.split("_")[1:])
            else:
                prefix = "Other"
                suffix = key

            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append((suffix, value))

    # Print grouped flat metrics
    for prefix, metrics in grouped.items():
        print(f"\n--- {prefix} Metrics ---")
        # Find max key length for alignment
        max_len = max(len(m[0]) for m in metrics)
        for suffix, value in metrics:
            if isinstance(value, float):
                print(f"  {suffix}:{' ' * (max_len - len(suffix) + 2)}{value:.6f}")
            else:
                print(f"  {suffix}:{' ' * (max_len - len(suffix) + 2)}{value}")

    # Print nested metrics (like Division_Frame Buffer)
    for key, value in nested.items():
        print(f"\n--- {key} ---")
        max_len = max(len(k) for k in value.keys())
        for metric_name, metric_value in value.items():
            if isinstance(metric_value, float):
                print(
                    f"  {metric_name}:{' ' * (max_len - len(metric_name) + 2)}{metric_value:.4f}"
                )
            else:
                print(
                    f"  {metric_name}:{' ' * (max_len - len(metric_name) + 2)}{metric_value}"
                )

    print("\n" + "=" * 60)
