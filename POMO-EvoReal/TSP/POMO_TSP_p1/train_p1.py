# -*- coding: utf-8 -*-
import os
import json
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from POMO.utils import *
from POMO.train_longTrain import train as TRAIN
from POMO.eval_longTrain import eval as EVAL


# Set log output
logging.basicConfig(level=logging.INFO)
CUDA_DEVICE_NUM = 0
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    # === Set the project path to "./tspdata_evo_main" ===
    ROOT_DIR = Path(__file__).resolve().parent
    os.chdir(ROOT_DIR)
    
    # experiment id
    idx_iteration = 0
    idx_response_id = datetime.now().strftime('%Y%m%d')
    # === path config ===
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_folder = f"POMO/result/checkpoints/checkpoint_{time_stamp}"
    os.makedirs(checkpoint_folder,exist_ok=True)
    train_set_dir = "dataset/Gen_Train"
    stdout_file_path = f"stdout_test{idx_response_id}.json"

    # training hyperparameters
    tolerance = 100 # set a high tolerance (>=num_outer_epochs) to avoid early-stopping
    model_save_interval = 5 # evaluate the model at every 5 epochs
    eval_start_epoch = 1 # start evaluation at epoch 1*5 = 5 
    num_outer_epochs = 61  # number of outer epoch, finetuning the model for a total of (61-1) * 5 = 300 epochs 


    # Weighted score function
    def weighted_gap_score(R, RC, C):
        return (19 * R + 12 * RC + 17 * C)/48

    # === Perform multiple TRAIN + EVAL ===
    results_per_epoch = {}
    patience_counter = 0
    best_score = float('inf')  # Lower score is better

    with open(stdout_file_path, "w") as f:
        try:
            init_gaps = {"R_aug_gap": 0.2221, "RC_aug_gap": 0.0988, "C_aug_gap": 0.0254}  # gaps are calculated by pre-trained model, for plotting graphs only
            

            gen_weights = [17,19,12] # [C, R, RC] or [S3, S1, S2]
            results_per_epoch["epoch_0"] = init_gaps


            for epoch in range(1, num_outer_epochs):  
            
                logger.info(f"[TEST] Running TRAIN for outer epoch {epoch}...")
                TRAIN(checkpoint_folder, idx_iteration, idx_response_id, CUDA_DEVICE_NUM, module_type="mixed", epoch=epoch, ratio=gen_weights,model_save_interval = model_save_interval)
                
                if epoch>=eval_start_epoch:
                  logger.info(f"[TEST] Running EVAL for outer epoch {epoch}...")
                  result = EVAL(train_set_dir, checkpoint_folder, idx_iteration, idx_response_id, CUDA_DEVICE_NUM, module_type="mixed", epoch=epoch)

                  label_dict = get_label()
                  type_gap_dict = compute_aug_gap_per_type(result, label_dict)
  
                  # extract the aug gaps for each type
                  R_aug_gap = type_gap_dict.get("R", 0.0)
                  RC_aug_gap = type_gap_dict.get("RC", 0.0)
                  C_aug_gap = type_gap_dict.get("C", 0.0)
  
                  result_summary = {
                      "R_aug_gap": round(R_aug_gap, 6),
                      "RC_aug_gap": round(RC_aug_gap, 6),
                      "C_aug_gap": round(C_aug_gap, 6)
                  }
                  logger.info(f"[TEST] Epoch {epoch} - R_aug_gap: {result_summary['R_aug_gap']}, RC_aug_gap: {result_summary['RC_aug_gap']}, C_aug_gap: {result_summary['C_aug_gap']}")
                  results_per_epoch[f"epoch_{epoch}"] = result_summary
  
                  # compute the weighted gap
                  current_score = weighted_gap_score(R_aug_gap, RC_aug_gap, C_aug_gap)
                  logger.info(f"[TEST] Weighted Avg gap at epoch {epoch}: {current_score:.6f}")
  
                  # Early stopping check
                  if current_score < best_score:
                      best_score = current_score
                      logger.info(f"[TEST] Current Best Score:{best_score:.6f}")
                      patience_counter = 0  # Reset patience if improved
                  else:
                      patience_counter += 1
                      logger.info(f"[TEST] No improvement. Patience counter: {patience_counter}")
  
                  if patience_counter >= tolerance:
                      logger.info(f"[TEST] Early stopping triggered at epoch {epoch}. Best score: {best_score:.6f}")
                      break

            json.dump(results_per_epoch, f, indent=2)

            import matplotlib.pyplot as plt
    
            epochs = list(range(eval_start_epoch-1, len(results_per_epoch)))
            R_gaps = [results_per_epoch[f"epoch_{e}"]["R_aug_gap"] for e in epochs]
            RC_gaps = [results_per_epoch[f"epoch_{e}"]["RC_aug_gap"] for e in epochs]
            C_gaps = [results_per_epoch[f"epoch_{e}"]["C_aug_gap"] for e in epochs]
            weighted_gaps = [weighted_gap_score(
                results_per_epoch[f"epoch_{e}"]["R_aug_gap"],
                results_per_epoch[f"epoch_{e}"]["RC_aug_gap"],
                results_per_epoch[f"epoch_{e}"]["C_aug_gap"]
            ) for e in epochs]
    
            plt.figure(figsize=(10,6))
            plt.plot(epochs, R_gaps, label='R_aug_gap', marker='o')
            plt.plot(epochs, RC_gaps, label='RC_aug_gap', marker='o')
            plt.plot(epochs, C_gaps, label='C_aug_gap', marker='o')
            plt.plot(epochs, weighted_gaps, label='Weighted_aug_gap', linestyle='--', marker='x')
    
            plt.xlabel('Epoch')
            plt.ylabel('Gap')
            plt.title('Aug Gap and Weighted Gap Over Epochs')
            plt.legend()
            plt.grid(True)
    
            save_path = os.path.join(ROOT_DIR, "aug_gap_plot.png")
            plt.savefig(save_path)
            logger.info(f"[TEST] Gap plot saved to {save_path}")
            plt.close()
            
            # Count Total Training Time
            end_time = time.time()
            total_time_sec = end_time - start_time
            hours, rem = divmod(total_time_sec, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            logger.info(f"[TEST] Total execution time: {time_str}")

        except Exception as e:
            f.write("Traceback: " + str(e) + '\n')
            logging.error(f"[TEST] Execution failed: {e}")

if __name__ == "__main__":
    main()

