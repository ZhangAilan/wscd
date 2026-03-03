

import os
import sys

import argparse
import numpy as np

import imageio

import torch



from core.networks import *
from core.WS_dataset import *

from tools.general.io_utils import *
from tools.general.time_utils import *


# from tools.ai.log_utils import *
from tools.ai.demo_utils import *

from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *

from accuray_metrics import calculate_metrics


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--image_size', default=256, type=int)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', required=True, type=str)
parser.add_argument('--domain', default='train', choices=['train', 'test'], type=str, help='Domain to process: train or test')
parser.add_argument('--crf_iteration', default=0, type=int)


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    cam_dir = f'./experiments/predictions/{args.experiment_name}_{args.domain}_npy/'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Process train or test dataset to find best threshold
    ###################################################################################
    dataset = WSCDDataSet_with_ID(pre_img_folder=args.data_dir+'/A', post_img_folder=args.data_dir+'/B',
                                       list_file=args.data_dir+'/list/'+args.domain+'_label.txt',
                                       img_size=args.image_size,change_only= False)

    # Define thresholds to iterate over
    thresholds = [i * 0.05 for i in range(1, 20)]  # 0.05 to 0.95 with step 0.05

    best_f1_score = -1
    best_threshold = None
    best_predictions_dir = None
    previous_f1_score = -1
    consecutive_decreases = 0
    max_consecutive_decreases = 2  # Stop after 2 consecutive decreases

    # Iterate through each threshold
    for i, threshold in enumerate(thresholds):
        print(f"Evaluating threshold: {threshold}")

        # Create temporary directory for current threshold predictions
        temp_pred_dir = create_directory(f'./tmp/predictions/{args.experiment_name}_{args.domain}@threshold{threshold}/')

        #################################################################################################
        # Generate pseudo labels for current threshold
        #################################################################################################
        with torch.no_grad():
            length = len(dataset)
            for step, (ori_imageA, ori_imageB, label, image_id) in enumerate(dataset):
                png_path = temp_pred_dir + image_id + '.png'
                if os.path.isfile(png_path):
                    continue

                ori_w, ori_h = ori_imageB.size
                predict_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()

                keys = predict_dict['keys']

                cams = predict_dict['hr_cam']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)

                cams = np.argmax(cams, axis=0)
                if keys.shape[0]>1:

                    if args.crf_iteration > 0:
                        cams = crf_inference_label(np.asarray(ori_imageB), cams, n_labels=keys.shape[0], t=args.crf_iteration)
                else:
                    pass

                conf = keys[cams]*255
                imageio.imwrite(png_path, conf.astype(np.uint8))

                sys.stdout.write('\r# Make Pseudo Labels [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
                sys.stdout.flush()

        # Calculate metrics for current threshold
        label_path = os.path.join(args.data_dir, 'label')  # Assuming label folder is in data_dir
        try:
            metrics = calculate_metrics(label_path, temp_pred_dir)
            current_f1_score = metrics.get('f1_score', 0)
            print(f"Threshold {threshold}: F1 Score = {current_f1_score}")

            # Update best threshold if current is better
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_threshold = threshold
                best_predictions_dir = temp_pred_dir
                consecutive_decreases = 0  # Reset counter when improvement is found
            else:
                # Check if F1 score is decreasing
                if previous_f1_score != -1 and current_f1_score < previous_f1_score:
                    consecutive_decreases += 1
                else:
                    consecutive_decreases = 0  # Reset if score increases or stays same

            # Early stopping condition
            if consecutive_decreases >= max_consecutive_decreases:
                print(f"F1 score has decreased for {consecutive_decreases} consecutive thresholds. Stopping early.")

                # Clean up remaining temporary directories that won't be used
                for j in range(i+1, len(thresholds)):
                    remaining_threshold = thresholds[j]
                    remaining_temp_dir = f'./tmp/predictions/{args.experiment_name}_{args.domain}@threshold{remaining_threshold}/'
                    if os.path.exists(remaining_temp_dir):
                        import shutil
                        shutil.rmtree(remaining_temp_dir)
                        print(f"Removed unused temporary directory: {remaining_temp_dir}")

                break

            previous_f1_score = current_f1_score
        except Exception as e:
            print(f"Error calculating metrics for threshold {threshold}: {str(e)}")
            continue

    print(f"\nBest threshold: {best_threshold} with F1 Score: {best_f1_score}")

    # Create final directory with best threshold and copy the best results there
    final_pred_dir = create_directory(f'./experiments/predictions/{args.experiment_name}_{args.domain}_pseudo_labels/')

    # Copy the best predictions to the final directory
    import shutil
    for filename in os.listdir(best_predictions_dir):
        if filename.endswith('.png'):
            src_path = os.path.join(best_predictions_dir, filename)
            dst_path = os.path.join(final_pred_dir, filename)
            shutil.copy2(src_path, dst_path)

    # Save best accuracy information to a txt file
    output_filename = f"{args.experiment_name}_{args.domain}_pseudo_labels@threshold{best_threshold}.txt"
    output_path = os.path.join('./experiments/predictions/', output_filename)

    # Calculate all metrics for the best threshold
    best_metrics = calculate_metrics(label_path, best_predictions_dir)

    # Write the results to the file
    with open(output_path, 'w') as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"CRF Iterations: {args.crf_iteration}\n")
        f.write(f"Best Threshold: {best_threshold}\n")
        f.write(f"Best F1 Score: {best_f1_score}\n")
        for metric_name, metric_value in best_metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")

    print(f"Results saved to {output_path}")

    # Clean up temporary directories
    import shutil
    tmp_dir = './tmp/predictions/'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    print(f"Final pseudo labels saved with best threshold {best_threshold}")