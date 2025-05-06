import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess
import torch
import torch.nn.functional as F
from torchvision.io.image import read_image, write_png
from torchvision.transforms.functional import rotate
import wfdb
import pandas as pd

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class ECGDigitizer:
    """
    ECG Digitizer class for converting ECG images to .hea and .dat files
    """
    def __init__(self, 
                 model_folder="ECG-Digitiser/models/M3/",
                 dataset_name="Dataset500_Signals",
                 image_type="png",
                 frequency=500,
                 long_signal_length_sec=10,
                 short_signal_length_sec=2.5,
                 signal_units="mV",
                 fmt='16',
                 adc_gain=1000.0,
                 baseline=0,
                 verbose=False,
                 show_image=False):
        """
        Initialize the ECG Digitizer
        
        Args:
            model_folder: Path to the nnU-Net model folder
            dataset_name: Name of the dataset for the nnU-Net model
            image_type: Type of input image (png, jpg, etc.)
            frequency: Sampling frequency of the signals
            long_signal_length_sec: Length in seconds of the full signal
            short_signal_length_sec: Length in seconds of the cropped signal
            signal_units: Units of the signal for the y-axis
            fmt: Format of the signal
            adc_gain: ADC gain of the signal
            baseline: Baseline of the signal
            verbose: Whether to print verbose output
            show_image: Whether to show the image with the mask
        """
        self.model_folder = model_folder
        self.dataset_name = dataset_name
        self.image_type = image_type
        self.frequency = frequency
        self.long_signal_length_sec = long_signal_length_sec
        self.short_signal_length_sec = short_signal_length_sec
        self.signal_units = signal_units
        self.fmt = fmt
        self.adc_gain = adc_gain
        self.baseline = baseline
        self.verbose = verbose
        self.show_image = show_image
        
        # Define lead label mapping
        self.lead_label_mapping = {
            "I": 1,
            "II": 2,
            "III": 3,
            "aVR": 4,
            "aVL": 5,
            "aVF": 6,
            "V1": 7,
            "V2": 8,
            "V3": 9,
            "V4": 10,
            "V5": 11,
            "V6": 12,
        }
        
        # Define y-shift ratio for each lead
        self.y_shift_ratio = {
            "I": 12.6 / 21.59,
            "II": 9 / 21.59,
            "III": 5.4 / 21.59,
            "aVR": 12.6 / 21.59,
            "aVL": 9 / 21.59,
            "aVF": 5.4 / 21.59,
            "V1": 12.59 / 21.59,
            "V2": 9 / 21.59,
            "V3": 5.4 / 21.59,
            "V4": 12.59 / 21.59,
            "V5": 9 / 21.59,
            "V6": 5.4 / 21.59,
            "full": 2.1 / 21.59,
        }

    def get_rotation_angle(self, np_image):
        """Get the rotation angle of the image."""
        lines = self._get_lines(np_image, threshold_HoughLines=1200)
        filtered_lines = self._filter_lines(
            lines, degree_window=30, parallelism_count=3, parallelism_window=2
        )
        if filtered_lines is None:
            rot_angle = np.nan
        else:
            rot_angle = self._get_median_degrees(filtered_lines)
        return rot_angle

    def _get_median_degrees(self, lines):
        """Get the median angle of the lines."""
        lines = lines[:, 0, :]
        line_angles = [-(90 - line[1] * 180 / np.pi) for line in lines]
        return round(np.median(line_angles), 4)

    def _is_within_x_degrees_of_horizontal(self, theta, degree_window):
        """Check if the line is within x degrees of horizontal (90 degrees)."""
        theta_degrees = theta * 180 / np.pi
        deviation_from_horizontal = abs(90 - theta_degrees)
        return deviation_from_horizontal < degree_window

    def _get_lines(self, np_image, threshold_HoughLines=1380, rho_resolution=1):
        """Get the lines in the image."""
        # Convert the image to a grayscale NumPy array
        image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the Canny edge detector to find edges in the image
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

        # Use HoughLines to find lines in the edge-detected image
        lines = cv2.HoughLines(
            edges, rho_resolution, np.pi / 180, threshold_HoughLines, None, 0, 0
        )

        return lines

    def _filter_lines(self, lines, degree_window=20, parallelism_count=0, parallelism_window=2):
        """Filter the lines to get the rotation angle."""
        parallelism_radian = np.deg2rad(parallelism_window)
        filtered_lines = []
        line_angles = []

        # Filter lines to be within the degree window of horizontal
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    if self._is_within_x_degrees_of_horizontal(theta, degree_window):
                        filtered_lines.append((rho, theta))
                        line_angles.append(theta)

        # Further filter lines based on parallelism
        parallel_lines = []
        if len(filtered_lines) > 0:
            for rho, theta in filtered_lines:
                count = 0
                for comp_rho, comp_theta in filtered_lines:
                    if (
                        abs(theta - comp_theta) < parallelism_radian
                        or abs((theta - comp_theta) - np.pi) < parallelism_radian
                    ):
                        count += 1
                if count >= parallelism_count:
                    parallel_lines.append((rho, theta))

        if len(parallel_lines) == 0:
            parallel_lines = None
        else:
            parallel_lines = np.array(parallel_lines)[:, np.newaxis, :]

        return parallel_lines

    def predict_mask_nnunet(self, image, temp_folder="data/temp"):
        """Predict the mask using nnUNet."""
        # Define temporary folders and paths
        temp_folder_input = f"{temp_folder}/temp_nnUNet_input"
        temp_folder_output = f"{temp_folder}/temp_nnUNet_output"
        image_path_temp = os.path.join(temp_folder_input, "00000_temp_0000.png")
        mask_path_temp = os.path.join(temp_folder_output, "00000_temp.png")

        # Create temp folders
        os.makedirs(temp_folder_input, exist_ok=True)
        os.makedirs(temp_folder_output, exist_ok=True)
        
        # Write image to temp folder
        write_png(image, image_path_temp)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=device,
                verbose=True
            )

        predictor.initialize_from_trained_model_folder(
                os.path.join(os.getcwd(),self.model_folder, "nnUNet_results", self.dataset_name, "nnUNetTrainer__nnUNetPlans__2d"),
                use_folds=['all'],
                checkpoint_name="checkpoint_final.pth",
            )
        if self.verbose:
            print(f"Predicting mask on {image_path_temp} with nnUNet")

        predictor.predict_from_files(
                [[str(image_path_temp)]],
                temp_folder_output,
                save_probabilities=False,
                overwrite=True
            )
        
        
        mask = read_image(mask_path_temp)

        # Delete all temporary folders and files
        shutil.rmtree(temp_folder_input, ignore_errors=True)
        shutil.rmtree(temp_folder_output, ignore_errors=True)

        return mask

    def cut_to_mask(self, img, mask, return_y1=False):
        """Cut the image to the mask."""
        coords = torch.where(mask[0] >= 1)
        y_min, y_max = coords[0].min().item(), coords[0].max().item()
        x_min, x_max = coords[1].min().item(), coords[1].max().item()
        img = img[:, y_min : y_max + 1, x_min : x_max + 1]
        if return_y1:
            return img, y_min, x_min
        else:
            return img

    def cut_binary(self, mask_to_use, image_rotated):
        """Cut the binary mask into single binary masks."""
        import pandas as pd
        
        signal_masks = {}
        signal_images = {}
        signal_positions = {}
        mask_values = list(pd.Series(mask_to_use.numpy().flatten()).value_counts().index)
        lead_names_in_mask = self.lead_label_mapping
        
        for lead_name, lead_value in lead_names_in_mask.items():
            binary_mask = torch.where(mask_to_use == lead_value, 1, 0)
            if binary_mask.sum() > 0:
                signal_img, y1, x1 = self.cut_to_mask(image_rotated, binary_mask, True)
                signal_mask = self.cut_to_mask(binary_mask, binary_mask)
                signal_images[lead_name] = signal_img
                signal_masks[lead_name] = signal_mask
                signal_positions[lead_name] = {"y1": y1, "x1": x1}
            else:
                signal_images[lead_name] = None
                signal_masks[lead_name] = None
                signal_positions[lead_name] = None

        return signal_masks, signal_positions, signal_images

    def vectorise(self, image_rotated, mask, signal_cropped, sec_per_pixel, mV_per_pixel, lead):
        """Vectorise the image."""
        # Get scaling info
        total_seconds_from_mask = round(torch.tensor(sec_per_pixel).item() * mask.shape[2], 1)
        if total_seconds_from_mask > (self.long_signal_length_sec / 2):
            total_seconds = self.long_signal_length_sec
            y_shift_ratio_ = self.y_shift_ratio["full"]
        else:
            total_seconds = self.short_signal_length_sec
            y_shift_ratio_ = self.y_shift_ratio[lead]
        values_needed = int(total_seconds * self.frequency)

        # Scale y
        non_zero_mean = torch.tensor(
            [
                torch.mean(torch.nonzero(mask[0, :, i]).type(torch.float32))
                for i in range(mask.shape[2])
            ]
        )
        signal_cropped_shifted = (1 - y_shift_ratio_) * image_rotated.shape[
            1
        ] - signal_cropped
        predicted_signal = (signal_cropped_shifted - non_zero_mean) * mV_per_pixel

        # Scale x
        n = predicted_signal.shape[0]
        data_reshaped = predicted_signal.view(1, 1, n)
        resampled_data = F.interpolate(
            data_reshaped, size=values_needed, mode="linear", align_corners=False
        )
        predicted_signal_sampled = resampled_data.view(-1)

        return predicted_signal_sampled

    def save_plot_masks_and_signals(self, image, masks_cropped, mask_start_position, signals, sig_names, output_folder, filename="record.png"):
        """Save a plot of the masks and signals."""
        num_signals = signals.shape[1]
        fig, axs = plt.subplots(
            1 + num_signals, 1, 
            figsize=(10, 2.5 * (1 + num_signals)),
            gridspec_kw={'height_ratios': [4] + [1] * num_signals}
        )

        if hasattr(image, "numpy"):
            image = image.numpy()
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        if image.ndim == 3 and image.shape[0] in [3, 4]:
            image = image.transpose(1, 2, 0)

        mask_combined = np.zeros_like(image, dtype=np.uint8) if image.ndim == 2 else np.zeros(image.shape[:2], dtype=np.uint8)
        for lead, mask_cropped in masks_cropped.items():
            if mask_cropped is not None:
                if mask_cropped.ndim == 3 and mask_cropped.shape[0] == 1:
                    mask_cropped = mask_cropped.squeeze(0)
                start_row = mask_start_position[lead]["y1"]
                start_col = mask_start_position[lead]["x1"]
                mask_height, mask_width = mask_cropped.shape
                mask_combined[start_row:start_row + mask_height, start_col:start_col + mask_width] = np.maximum(
                    mask_combined[start_row:start_row + mask_height, start_col:start_col + mask_width],
                    mask_cropped
                )

        axs[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axs[0].imshow(mask_combined, cmap="jet", alpha=0.5)
        axs[0].set_title("Masks overlayed on image")
        axs[0].axis("off")

        time_axis = np.arange(signals.shape[0])
        for i, signal in enumerate(signals.T):
            axs[i + 1].plot(time_axis, signal)
            axs[i + 1].set_title(sig_names[i])
            axs[i + 1].set_xlabel("Time")
            axs[i + 1].set_ylabel("Signal amplitude")
            axs[i + 1].grid()

        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, filename), dpi=300)
        plt.close(fig)

    def digitize(self, image_path, output_folder=None, temp_folder="data/temp", cleanup=True):
        """
        Digitize an ECG image and save the .hea and .dat files
        
        Args:
            image_path: Path to the ECG image
            output_folder: Folder to save the output files
            temp_folder: Folder to save temporary files
            cleanup: Whether to clean up temporary files
            
        Returns:
            signals: Dictionary of signal data
            signal_names: List of signal names
        """
        if output_folder is None:
            output_folder = os.path.dirname(image_path)
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Get record name
        record = os.path.basename(image_path).replace(f".{self.image_type}", "")
        
        # Read image
        image = read_image(image_path)
        image = image[:3]  # Take only RGB channels

        # Rotate image
        rot_angle = self.get_rotation_angle(image.permute(1, 2, 0).numpy().astype(np.uint8))
        if np.isnan(rot_angle):
            if self.verbose:
                print(f"Warning: Could not determine rotation angle for {record}. Using 0 degrees.")
            rot_angle = 0
        image_rotated = rotate(image, rot_angle)

        # Segment image
        mask_to_use = self.predict_mask_nnunet(image_rotated, temp_folder)

        # Use mask to cut into single, binary masks
        signal_masks_cropped, signal_positions_cropped, _ = self.cut_binary(
            mask_to_use, image_rotated
        )

        # Vectorize
        x_pixel_list = [
            v.shape[2] for v in signal_masks_cropped.values() if v is not None
        ]
        if len(x_pixel_list) == 0:
            if self.verbose:
                print(f"Warning: No signals found in {record}")
            return None, None
        
        x_pixel_list_median = np.median(x_pixel_list)
        x_pixel_list_below_2x_median_mean = np.mean(
            [v for v in x_pixel_list if v < 2 * x_pixel_list_median]
        )
        sec_per_pixel = 2.5 / x_pixel_list_below_2x_median_mean
        mm_per_pixel = 25 * sec_per_pixel
        sec_per_pixel = mm_per_pixel / 25
        mV_per_pixel = mm_per_pixel / 10
        signals_predicted = {}
        
        for lead, mask in signal_masks_cropped.items():
            if mask is not None:
                signals_predicted[lead] = self.vectorise(
                    image_rotated,
                    mask,
                    signal_positions_cropped[lead]["y1"],
                    sec_per_pixel,
                    mV_per_pixel,
                    lead,
                )
            else:
                signals_predicted[lead] = None

        # Save Challenge outputs.
        signals = {
            signal_name: signals_predicted[signal_name].numpy()
            for signal_name in self.lead_label_mapping.keys()
            if signals_predicted[signal_name] is not None
        }
        
        num_samples = int(self.long_signal_length_sec * self.frequency)
        signal_list = []
        for signal in signals.values():
            if len(signal) < num_samples:
                nan_signal = np.empty(num_samples)
                nan_signal[:] = np.nan
                nan_signal[: int(len(signal))] = signal
                signal_list.append(nan_signal)
            else:
                signal_list.append(signal)
        
        sig_names = list(signals.keys())
        signals_array = np.array(signal_list).T

        if self.show_image:
            if self.verbose:
                print(f"Storing image of shape {image_rotated.shape}")
            self.save_plot_masks_and_signals(
                image_rotated,
                signal_masks_cropped,
                signal_positions_cropped,
                signals_array,
                sig_names,
                output_folder,
                f"{record}.png",
            )

        if self.verbose:
            print(f"Storing signals for record {record} with shape {signals_array.shape}")
            
        # Normalize if signal is out of bounds
        if (np.nanmax(signals_array) > 10) or (np.nanmin(signals_array) < -10):
            if self.verbose:
                print(f"Signal out of range for record {record}, normalizing to range between 1 and -1")
            max_val = np.nanmax(signals_array)
            min_val = np.nanmin(signals_array)
            signals_array = (signals_array - min_val) / (max_val - min_val) * 2 - 1
            
        # Write WFDB files
        wfdb.wrsamp(
            record,
            fs=self.frequency,
            units=[self.signal_units] * signals_array.shape[1],
            sig_name=sig_names,
            p_signal=np.nan_to_num(signals_array),
            write_dir=output_folder,
            fmt=[self.fmt] * signals_array.shape[1],
            adc_gain=[self.adc_gain] * signals_array.shape[1],
            baseline=[self.baseline] * signals_array.shape[1],
        )
        
        # Cleanup temporary files
        if cleanup:
            import shutil
            temp_folder_input = f"{temp_folder}/temp_nnUNet_input"
            temp_folder_output = f"{temp_folder}/temp_nnUNet_output"
            if os.path.exists(temp_folder_input):
                shutil.rmtree(temp_folder_input)
            if os.path.exists(temp_folder_output):
                shutil.rmtree(temp_folder_output)
        
        return signals, sig_names

    def digitize_folder(self, input_folder, output_folder=None, temp_folder="data/temp", cleanup=True):
        """
        Digitize all ECG images in a folder
        
        Args:
            input_folder: Folder containing ECG images
            output_folder: Folder to save the output files
            temp_folder: Folder to save temporary files
            cleanup: Whether to clean up temporary files
            
        Returns:
            results: Dictionary of results for each image
        """
        if output_folder is None:
            output_folder = input_folder
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all images
        from tqdm import tqdm
        
        image_files = [
            f for f in os.listdir(input_folder) if f.endswith(f".{self.image_type}")
        ]
        
        results = {}
        for image_file in tqdm(image_files, desc="Digitizing ECG images"):
            image_path = os.path.join(input_folder, image_file)
            record = image_file.replace(f".{self.image_type}", "")
            
            signals, sig_names = self.digitize(
                image_path, 
                output_folder=output_folder,
                temp_folder=temp_folder,
                cleanup=False  # We'll clean up at the end
            )
            
            results[record] = {
                "signals": signals,
                "signal_names": sig_names
            }
            
        # Cleanup temporary files
        if cleanup:
            import shutil
            temp_folder_input = f"{temp_folder}/temp_nnUNet_input"
            temp_folder_output = f"{temp_folder}/temp_nnUNet_output"
            if os.path.exists(temp_folder_input):
                shutil.rmtree(temp_folder_input)
            if os.path.exists(temp_folder_output):
                shutil.rmtree(temp_folder_output)
                
        return results