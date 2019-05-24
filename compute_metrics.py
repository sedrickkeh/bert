def compute_metrics(preds, labels):
    return (preds == labels).mean()

