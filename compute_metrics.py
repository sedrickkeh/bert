def compute_metrics(preds, labels):
    return {"acc": (preds == labels).mean()}

