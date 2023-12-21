from torchmetrics.classification import BinaryROC

# Set up the Metric, 21 means 0.05 steps per threshold
roc_curve = BinaryROC(thresholds=21)

"""
        test_features, test_labels = next(iter(testing_dataloader))
        # Send to GPU
        data = test_features.to(device)
        start = time.process_time_ns()
        output = model(data)
        t_infer = time.process_time_ns() - start
        # Fetch from GPU
        output = output.cpu().detach()
        test_features = test_features.cpu().detach()
        # ROC
        fpr, tpr, thresholds = roc_curve(output, test_labels.squeeze().to(torch.int))
        idx = find_fpr_index(fpr, fpr_limit)
        plot_roc_curve(fpr, tpr, thres_idx=idx, display=True)
        # Scatter & Comparison
        sample_img = create_sample_comparison(test_features, output, test_labels, t_infer=t_infer,
                                              threshold=thresholds.numpy()[idx])
        show_data_pair(sample_img)
"""
for batch_idx, (samples, targets) in enumerate(testing_dataloader):
    # send data to GPU
    samples = samples.to(device)
    start = time.process_time_ns()
    output = model(samples)
    t_infer = time.process_time_ns() - start
    # fetch data from GPU
    output = output.cpu().detach()
    # Attach to ROC curve
    fpr, tpr, thresholds = roc_curve(output, targets.squeeze().to(torch.int))
    # https://github.com/wandb/wandb/issues/1076
    # wandb_logger.log({"roc": wandb.plot.roc_curve(targets, output)})

    if batch_idx % 100 == 0:
        # Create Image Sample
        samples = samples.cpu().detach()
        idx = find_fpr_index(fpr, fpr_limit)
        threshold = thresholds.numpy()[idx]
        sample_img = create_sample_comparison(samples, output, targets, t_infer=t_infer,
                                              threshold=threshold)
        wandb_image = wandb.Image(sample_img,
                                  caption=f"Batch-ID= {batch_idx}\nTop: Output\nBottom: Target")
        wandb_logger.log({"Examples": wandb_image})
        # Create ROC snippet
        sample_roc = plot_roc_curve(fpr, tpr, thres_idx=idx)
        sample_auc = np.round(metrics.auc(fpr, tpr), decimals=3)
        wandb_roc = wandb.Image(sample_roc,
                                caption=f"Batch-ID= {batch_idx}\nROC with {threshold}\nAUC: {sample_auc}")
        wandb_logger.log({"Examples ROC": wandb_roc})

fpr, tpr, thresholds = roc_curve.compute()
# plot_roc_curve(fpr, tpr, thres_idx=find_threshold_index(thresholds, threshold), display=True)

data = [[x, y] for (x, y) in zip(fpr, tpr)]
table = wandb.Table(data=data, columns=["False Positive Rate", "True Positive Rate"])
wandb_logger.log({"ROC curve": wandb.plot.line(table, "True Positive Rate", "False Positive Rate",
                                               title=f"ROC Curve over {testing_data.__len__()} samples")})
wandb_logger.log({"AUC": metrics.auc(fpr, tpr)})



def find_fpr_index(fpr_array, fpr):
    fpr_array = fpr_array.numpy()
    fpr_idx = np.where(fpr_array >= fpr)
    return fpr_idx[0][0]

def find_threshold_index(thres_array, threshold):
    thres_array = thres_array.numpy().round(decimals=2)
    threshold_idx = np.where(thres_array == threshold)
    return threshold_idx

