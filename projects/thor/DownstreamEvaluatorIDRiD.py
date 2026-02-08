import logging
import io
import os
import csv
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import seaborn as sns
from torch.nn import L1Loss
from model_zoo.vgg import VGGEncoder
import lpips
from core.DownstreamEvaluator import DownstreamEvaluator
import numpy as np
import torch
import cv2
import copy
import sklearn.metrics 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    IDRiD Downstream Evaluator (Local Save Version)
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True, threshold_percentile=0.95, val_data_dict=None):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = torch.nn.MSELoss().to(self.device)
        self.compute_scores = True
        
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        self.threshold_percentile = threshold_percentile
        self.val_data_dict = val_data_dict

        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_dir = os.path.join("results", current_time)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.viz_dir = os.path.join(self.save_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

        logging.info(f"######## Evaluation Results will be saved to: {self.save_dir} ########")

    def start_task(self, global_model):
        # THOR threshold default
        th = 0.1
        self.object_localization(global_model, th)

    def _to_slash_path(self, p: str) -> str:
        return (p or "").replace("\\", "/")

    def object_localization(self, global_model, th=0):
        logging.info(f"################ Object Localization TEST (IDRiD) TH: {th} #################")
        
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()

        score_threshold = None
        if self.val_data_dict:
            logging.info("Computing threshold from validation normal data")
            val_scores, val_labels = self._collect_scores(self.val_data_dict)
            normal_scores = val_scores[val_labels == 0]
            if normal_scores.size > 0:
                score_threshold = float(np.percentile(normal_scores, self.threshold_percentile * 100.0))
                logging.info(f"Validation threshold (Normal P{self.threshold_percentile*100:.1f}): {score_threshold:.6f}")
            else:
                logging.warning("No normal samples found in validation data for thresholding.")

        # Metrics storage
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'Anomaly_Score': [], 
            'Label': []          
        }
        
        csv_rows = []
        csv_header = ['Dataset', 'Filename', 'Label', 'Anomaly_Score', 'MSE', 'LPIPS']

        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info(f'DATASET: {dataset_key}')
            
            for idx, data in enumerate(dataset):
                if isinstance(data, dict) and 'images' in data.keys():
                    data0 = data['images']
                    filenames = data.get('filenames', [f'{dataset_key}_{idx}'])
                elif isinstance(data, (list, tuple)):
                    data0 = data[0]
                    filenames = data[2] if len(data) >= 3 else [f'{dataset_key}_{idx}'] #changed 2 to 3
                else:
                    data0 = data
                    filenames = [f'{dataset_key}_{idx}']

                x = data0.to(self.device)
                
                if len(x.shape) == 5: 
                    x = x.squeeze(2)
                
                #[0, 1] -> [-1, 1]
                x_input = (x * 2) - 1

                with torch.no_grad():
                    anomaly_maps, anomaly_scores, x_rec_dict = self.model.get_anomaly(x_input)

                    x_rec = x_rec_dict['x_rec']
                    #[-1, 1] -> [0, 1]
                    x_rec = (x_rec + 1) / 2
                    x_rec = torch.clamp(x_rec, 0, 1)

                    for i in range(len(x)):
                        count = str(idx * len(x) + i)
                        filename = filenames[i] if i < len(filenames) else f"img_{count}"

                        fn = self._to_slash_path(filename) if hasattr(self, "_to_slash_path") else (filename or "").replace("\\", "/")
                        current_label = 0 if ("/normal/" in fn or "/good/" in fn) else 1

                        x_i = x[i]      
                        x_rec_i = x_rec[i] 
                        x_res_i = anomaly_maps[i][0] 
                        img_np = x_i.detach().cpu().numpy()

                        if img_np.ndim == 3 and img_np.shape[0] == 3:   
                            g = img_np[1]
                        else:                                         
                            g = img_np[..., 1]

                        g8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
                        g8 = cv2.GaussianBlur(g8, (5, 5), 0)
                        _, fov = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # MORPHOLOGY
                        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                        fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, k_close)
                        fov = cv2.morphologyEx(fov, cv2.MORPH_OPEN,  k_open)

                        num, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
                        if num > 1:
                            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            fov = (labels == largest).astype(np.uint8) * 255
                        

                        fov01 = self._inner_ellipse_mask(fov, shrink_px=10)  # try 20..40

                        mask_u8 = (fov01 > 0).astype(np.uint8)
                        dist2 = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                        rim = 4
                        fov01 = (dist2 > rim).astype(np.float32)

                        # KERNEL METHOD
                        map_np = x_res_i.detach().cpu().numpy() if isinstance(x_res_i, torch.Tensor) else x_res_i
                        if map_np.ndim == 3 and map_np.shape[0] == 1:
                            map_np = map_np[0]

                        map_raw = np.clip(map_np, 0.0, 1.0)
                        map_np = map_raw * fov01

                        mask01 = fov01  
                        rec_np = x_rec_i.detach().cpu().numpy()  
                        rec_np = rec_np * mask01             

                        mask_t = torch.from_numpy(fov01).to(self.device).float()     
                        mask_t = mask_t.unsqueeze(0).repeat(3, 1, 1)               
                        den = mask_t.sum().clamp(min=1.0)
                        diff = (x_rec_i - x_i) * mask_t
                        loss_mse = (diff * diff).sum() / den
                        #loss_mse = self.criterion_rec(x_rec_i, x_i)
                        metrics['MSE'].append(loss_mse.item())
                        
                        # LPIPS Mean
                        x_i_masked = x_i * mask_t
                        x_rec_i_masked = x_rec_i * mask_t
                        loss_lpips = self.l_pips_sq(x_i_masked.unsqueeze(0), x_rec_i_masked.unsqueeze(0)).mean().item()
                        #loss_lpips = self.l_pips_sq(x_i.unsqueeze(0), x_rec_i.unsqueeze(0)).mean().item()
                        metrics['LPIPS'].append(loss_lpips)

                        # Score aligned with training objective (masked L2 reconstruction error)
                        score = loss_mse.item()
                        logging.info(
                        f"Score | dataset={dataset_key} | file={os.path.basename(filename)} | "
                        f"label={current_label} | score={score:.6f}")
                        

                        metrics['Anomaly_Score'].append(score)
                        metrics['Label'].append(current_label)

                        # CSV 
                        csv_rows.append([dataset_key, filename, current_label, score, loss_mse.item(), loss_lpips])
                        self._visualize_idrid(x_i, rec_np, map_np, filename, dataset_key, count, th, fov01)

            logging.info(f"Finished processing {dataset_key}")

        # Compute score threshold (fallback to test normals if no validation threshold)
        score_threshold = self._compute_global_metrics(metrics, score_threshold)
        if score_threshold is not None:
            logging.info(f"Score threshold (Normal P{self.threshold_percentile*100:.1f}): {score_threshold:.6f}")
            csv_header.extend(['Pred_Label', 'Correct'])
            for row in csv_rows:
                score = row[3]
                pred = 1 if score >= score_threshold else 0
                row.extend([pred, int(pred == row[2])])

        # CSV
        csv_path = os.path.join(self.save_dir, 'per_image_scores.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_rows)
        logging.info(f"Saved per-image scores to {csv_path}")

    def _collect_scores(self, data_dict):
        scores = []
        labels = []

        for dataset_key in data_dict.keys():
            dataset = data_dict[dataset_key]
            logging.info(f'VAL DATASET: {dataset_key}')

            for idx, data in enumerate(dataset):
                if isinstance(data, dict) and 'images' in data.keys():
                    data0 = data['images']
                    filenames = data.get('filenames', [f'{dataset_key}_{idx}'])
                elif isinstance(data, (list, tuple)):
                    data0 = data[0]
                    filenames = data[2] if len(data) >= 3 else [f'{dataset_key}_{idx}']
                else:
                    data0 = data
                    filenames = [f'{dataset_key}_{idx}']

                x = data0.to(self.device)
                if len(x.shape) == 5:
                    x = x.squeeze(2)

                x_input = (x * 2) - 1
                with torch.no_grad():
                    anomaly_maps, _, x_rec_dict = self.model.get_anomaly(x_input)

                for i in range(len(x)):
                    filename = filenames[i] if i < len(filenames) else f"img_{idx * len(x) + i}"
                    fn = self._to_slash_path(filename) if hasattr(self, "_to_slash_path") else (filename or "").replace("\\", "/")
                    current_label = 0 if ("/normal/" in fn or "/good/" in fn) else 1

                    x_i = x[i]
                    x_res_i = anomaly_maps[i][0]
                    img_np = x_i.detach().cpu().numpy()

                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        g = img_np[1]
                    else:
                        g = img_np[..., 1]

                    g8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
                    g8 = cv2.GaussianBlur(g8, (5, 5), 0)
                    _, fov = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, k_close)
                    fov = cv2.morphologyEx(fov, cv2.MORPH_OPEN, k_open)

                    num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
                    if num > 1:
                        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        fov = (labels_cc == largest).astype(np.uint8) * 255

                    fov01 = self._inner_ellipse_mask(fov, shrink_px=25)

                    mask_u8 = (fov01 > 0).astype(np.uint8)
                    dist2 = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                    rim = 8
                    fov01 = (dist2 > rim).astype(np.float32)

                    map_np = x_res_i.detach().cpu().numpy() if isinstance(x_res_i, torch.Tensor) else x_res_i
                    if map_np.ndim == 3 and map_np.shape[0] == 1:
                        map_np = map_np[0]

                    map_raw = np.clip(map_np, 0.0, 1.0)
                    map_np = map_raw * fov01

                    # Score aligned with training objective (masked L2 reconstruction error)
                    x_rec = x_rec_dict['x_rec']
                    x_rec = (x_rec + 1) / 2
                    x_rec = torch.clamp(x_rec, 0, 1)
                    x_rec_i = x_rec[i]
                    mask_t = torch.from_numpy(fov01).to(self.device).float()
                    mask_t = mask_t.unsqueeze(0).repeat(3, 1, 1)
                    den = mask_t.sum().clamp(min=1.0)
                    diff = (x_rec_i - x_i) * mask_t
                    loss_mse = (diff * diff).sum() / den
                    score = loss_mse.item()
                    scores.append(score)
                    labels.append(current_label)

        return np.array(scores), np.array(labels)


    def _inner_ellipse_mask(self, fov_u8: np.ndarray, shrink_px: int = 25) -> np.ndarray:
        f = (fov_u8 > 0).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return (f > 0).astype(np.float32)

        c = max(cnts, key=cv2.contourArea)
        if len(c) < 5 or cv2.contourArea(c) < 1000:
            return (f > 0).astype(np.float32)

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)   # full axis lengths
        MA2 = max(MA - 2 * shrink_px, 10.0)
        ma2 = max(ma - 2 * shrink_px, 10.0)

        mask = np.zeros_like(f, dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(round(cx)), int(round(cy))),
            (int(round(MA2 / 2.0)), int(round(ma2 / 2.0))),
            angle, 0, 360,
            255, -1
        )

        #mask = cv2.bitwise_and(mask, f)  # keep inside original FOV
        return (mask > 0).astype(np.float32)    

    def _visualize_idrid(self, x, x_rec, anomaly_map, filename, dataset_key, count, th, fov01=None):
        if isinstance(x, torch.Tensor):
            img_np = x.permute(1, 2, 0).cpu().detach().numpy()
        else:
            img_np = np.transpose(x, (1, 2, 0))

        if isinstance(x_rec, torch.Tensor):
            rec_np = x_rec.permute(1, 2, 0).cpu().detach().numpy()
        else:
            rec_np = np.transpose(x_rec, (1, 2, 0))
        
        img_np = np.clip(img_np, 0, 1)
        rec_np = np.clip(rec_np, 0, 1)

        # Apply FOV mask if available
        if fov01 is not None:
            mask3 = fov01[..., None]
            img_np = img_np * mask3
            rec_np = rec_np * mask3
        # END FOV mask application
 
        if isinstance(anomaly_map, torch.Tensor):
            map_np = anomaly_map.cpu().detach().numpy()
        else:
            map_np = anomaly_map
        
        if fov01 is not None:
            map_np = map_np * fov01  

        seg_np = (map_np >= th).astype(np.uint8) 
        if fov01 is not None:
            seg_np = seg_np * (fov01 > 0.5).astype(np.uint8)


        if fov01 is not None:
            m = (fov01 > 0.5)
            ys, xs = np.where(m)
            if ys.size > 0:
                pad = 5
                y0 = max(int(ys.min()) - pad, 0)
                y1 = min(int(ys.max()) + pad + 1, m.shape[0])
                x0 = max(int(xs.min()) - pad, 0)
                x1 = min(int(xs.max()) + pad + 1, m.shape[1])

                img_np = img_np[y0:y1, x0:x1, :]
                rec_np = rec_np[y0:y1, x0:x1, :]
                map_np = map_np[y0:y1, x0:x1]
                seg_np = seg_np[y0:y1, x0:x1]
            else:

                logging.warning(f"Empty FOV mask for {filename}, skipping crop")

        elements = [img_np, rec_np, map_np, seg_np]
        titles = ['Input (RGB)', 'Reconstruction', 'Anomaly Map', f'Pred > {th}']
        cmaps = [None, None, 'jet', 'gray'] 

        fig, axarr = plt.subplots(1, len(elements), figsize=(16, 4), gridspec_kw={'wspace': 0.05, 'hspace': 0})
        
        for idx, ax in enumerate(axarr):
            ax.axis('off')
            ax.set_title(titles[idx])
            if cmaps[idx] is None:
                ax.imshow(elements[idx]) 
            else:
                ax.imshow(elements[idx], cmap=cmaps[idx], vmin=0, vmax=1 if idx==3 else np.max(map_np))

        save_name = f"{os.path.basename(filename).split('.')[0]}.png"
        save_path = os.path.join(self.viz_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved visualization: {save_path}")


    def _compute_global_metrics(self, metrics, score_threshold=None):
        """
        Compute score threshold and save to local file
        """
        labels = np.array(metrics['Label'])
        scores = np.array(metrics['Anomaly_Score'])
        
        result_txt_path = os.path.join(self.save_dir, 'global_metrics.txt')

        lines = []
        score_threshold = None
        if len(np.unique(labels)) > 1:
            try:
                auroc = roc_auc_score(labels, scores)
                auprc = average_precision_score(labels, scores)
                log_msg = f"Global AUROC: {auroc:.4f}\nGlobal AUPRC: {auprc:.4f}"
                logging.info(log_msg)
                lines.append(f"Global AUROC: {auroc:.4f}")
                lines.append(f"Global AUPRC: {auprc:.4f}")

                fpr, tpr, thresholds = roc_curve(labels, scores)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")

                roc_path = os.path.join(self.save_dir, 'roc_curve.png')
                plt.savefig(roc_path)
                plt.close()
                logging.info(f"Saved ROC curve to {roc_path}")

            except Exception as e:
                logging.error(f"Failed to compute AUC/ROC: {e}")
        else:
            msg = "Cannot compute AUC: Only one class present in test set."
            logging.warning(msg)
            lines.append(msg)

        if score_threshold is None:
            normal_scores = scores[labels == 0]
            if normal_scores.size > 0:
                score_threshold = float(np.percentile(normal_scores, self.threshold_percentile * 100.0))
                lines.append(f"Score Threshold (Normal P{self.threshold_percentile*100:.1f}): {score_threshold:.6f}")
            else:
                logging.warning("No normal samples available for percentile-based threshold.")
        else:
            lines.append(f"Score Threshold (Normal P{self.threshold_percentile*100:.1f}): {score_threshold:.6f}")

        overlap = self._score_overlap(scores, labels)
        if overlap is not None:
            lines.append(f"Score Overlap (OVL): {overlap:.4f}")

        with open(result_txt_path, 'w') as f:
            f.write("\n".join(lines))

        return score_threshold

    def _score_overlap(self, scores: np.ndarray, labels: np.ndarray):
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        if normal_scores.size < 2 or anomaly_scores.size < 2:
            logging.warning("Not enough samples to compute overlap.")
            return None

        all_scores = np.concatenate([normal_scores, anomaly_scores])
        if np.allclose(all_scores.min(), all_scores.max()):
            logging.warning("Scores are constant; overlap is undefined.")
            return None

        bins = np.histogram_bin_edges(all_scores, bins='auto')
        if bins.size < 3:
            bins = np.linspace(all_scores.min(), all_scores.max(), 30)

        hist_n, _ = np.histogram(normal_scores, bins=bins, density=True)
        hist_a, _ = np.histogram(anomaly_scores, bins=bins, density=True)
        bin_widths = np.diff(bins)

        overlap = np.minimum(hist_n, hist_a)
        overlap_coeff = float(np.sum(overlap * bin_widths))

        plt.figure(figsize=(10, 6))
        plt.step(bins, np.r_[hist_n, hist_n[-1]], where='post', label='Normal', color='tab:blue')
        plt.step(bins, np.r_[hist_a, hist_a[-1]], where='post', label='Anomaly', color='tab:red')
        plt.bar(
            bins[:-1], overlap, width=bin_widths, align='edge',
            color='gray', alpha=0.35, label=f'Overlap (OVL={overlap_coeff:.3f})'
        )
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Score Distribution Overlap')
        plt.legend(loc='upper right')
        plt.text(
            0.02, 0.98,
            f"OVL={overlap_coeff:.3f}",
            transform=plt.gca().transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', alpha=0.7, ec='gray')
        )

        overlap_path = os.path.join(self.save_dir, 'score_overlap.png')
        plt.savefig(overlap_path)
        plt.close()
        logging.info(f"Saved overlap plot to {overlap_path}")

        return overlap_coeff
