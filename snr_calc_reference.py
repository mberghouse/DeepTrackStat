import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io, draw
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import time


def calculate_snr_with_reference(test_image, reference_image):
    """
    Calculate SNR using the formula:
    SNR = 10 * log10( signal_power / noise_power )
    where signal_power is the power of the reference image
    and noise_power is the power of the difference between reference and test
    """
    # Convert images to float (0 to 1 range)
    test_float = test_image.astype(np.float64) / 255.0
    ref_float = reference_image.astype(np.float64) / 255.0
    
    # Calculate signal power from reference image
    # Since particles are the signal (black = 0 in original image)
    signal = ref_float  # Invert so particles are 1.0 (signal) and background is 0.0
    signal_power = np.sum(signal**2)
    # contrast = np.max(signal) - np.min(signal)
    
    # Calculate noise as difference between reference and test
    test_inv = test_float#1.0 - test_float  # Invert test image the same way
    noise = signal - test_inv  # Difference between signal and test
    noise_power = np.sum(noise ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-10:
        return float('inf')
    
    # Calculate SNR in dB
    snr = 10*np.log10(signal_power / noise_power)
    
    return snr

def create_reference_and_augment_test(x_pos, y_pos, test_image, particle_diameter=2):
    """
    Create reference image and add reference particles to test image.
    
    Parameters:
        x_pos, y_pos: Sequences (or arrays) of x and y coordinates for particles
        test_image: Original test image to be augmented
        particle_diameter: Diameter of the particles in pixels
        
    Returns:
        Tuple (reference_image, augmented_test_image) of NumPy arrays
    """
    image_size = test_image.shape
    # Create white background for reference
    reference = np.ones(image_size, dtype=np.uint8) * 255
    # Create copy of test image for augmentation
    augmented_test = test_image.copy()
    
    radius = particle_diameter // 2

    # Handle NaN values in coordinates
    valid_coords = ~(np.isnan(x_pos) | np.isnan(y_pos))
    x_pos = x_pos[valid_coords]
    y_pos = y_pos[valid_coords]

    # Draw particles at each (x, y) coordinate
    for x, y in zip(x_pos, y_pos):
        x_int = int(round(float(x)))
        y_int = int(round(float(y)))
        
        if 0 <= x_int < image_size[1] and 0 <= y_int < image_size[0]:
            # Create particle mask
            rr, cc = draw.disk((y_int, x_int), radius, shape=image_size)
            
            # Add to reference image (black particles)
            reference[rr, cc] = 0
            
            # Add to test image (black particles)
            augmented_test[rr, cc] = 0
            
    return reference, augmented_test

def process_sequence(args):
    """Process a sequence of images for one simulation folder."""
    sim_folder, x_traj, y_traj = args
    
    def natural_sort_key(s):
        import re
        parts = re.split('([0-9]+)', str(s))
        parts[1::2] = map(int, parts[1::2])
        return parts

    image_files = sorted(Path(sim_folder).glob('*.jpg'), key=natural_sort_key)
    snr_values = []
    
    if len(image_files) == 0:
        print(f"No .jpg files found in {sim_folder}")
        return sim_folder.name, []
        
    if x_traj.shape[1] != y_traj.shape[1]:
        print(f"Trajectory arrays have mismatched dimensions: {x_traj.shape} vs {y_traj.shape}")
        return sim_folder.name, []
    
    for idx, img_path in enumerate(image_files):
        try:
            # Read original test image
            test_image = io.imread(img_path)
            #test_image=255-test_image
            
            if idx >= x_traj.shape[0]:
                print(f"Warning: No trajectory data for frame {idx}")
                continue
                
            # Create reference image and augmented test image
            ref_image, augmented_test = create_reference_and_augment_test(
                x_traj[idx],
                y_traj[idx],
                test_image,
                particle_diameter=10
            )
            
            # Calculate SNR for both original and augmented test images
            snr_original = calculate_snr_with_reference(test_image, ref_image)
            snr_augmented = calculate_snr_with_reference(augmented_test, ref_image)
            
            if np.isfinite(snr_original) and np.isfinite(snr_augmented):
                snr_values.append((str(img_path), snr_original, snr_augmented))
            else:
                print(f"Warning: Invalid SNR value for {img_path}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return sim_folder.name, snr_values

def save_intermediate_results(results, output_dir, iteration):
    """Save intermediate results to CSV files."""
    # Convert results to DataFrame
    data = []
    for sim_name, values in results:
        for img_path, snr_original, snr_augmented in values:
            data.append({
                'simulation': sim_name,
                'image_path': img_path,
                'snr_original': snr_original,
                'snr_augmented': snr_augmented
            })
    
    df = pd.DataFrame(data)
    
    # Compute statistics for both SNR types
    stats = {
        'mean_snr_original': df['snr_original'].mean(),
        'median_snr_original': df['snr_original'].median(),
        'std_snr_original': df['snr_original'].std(),
        'min_snr_original': df['snr_original'].min(),
        'max_snr_original': df['snr_original'].max(),
        'q25_snr_original': df['snr_original'].quantile(0.25),
        'q75_snr_original': df['snr_original'].quantile(0.75),
        'mean_snr_augmented': df['snr_augmented'].mean(),
        'median_snr_augmented': df['snr_augmented'].median(),
        'std_snr_augmented': df['snr_augmented'].std(),
        'min_snr_augmented': df['snr_augmented'].min(),
        'max_snr_augmented': df['snr_augmented'].max(),
        'q25_snr_augmented': df['snr_augmented'].quantile(0.25),
        'q75_snr_augmented': df['snr_augmented'].quantile(0.75),
    }
    
    # Save files
    df.to_csv(output_dir / f'snr_values_intermediate.csv', index=False)
    pd.Series(stats).to_csv(output_dir / f'snr_statistics_intermediate.csv')
    
    print(f"Saved intermediate results for iteration {iteration}")
    print("Current statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

def main():
    # Define paths
    sims_dir = Path("all_sims")
    traj_dir = Path("all_traj_files")
    output_dir = Path("snr_analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Prepare arguments for parallel processing
    process_args = []
    num_particles = []
    for sim_folder in sorted(sims_dir.iterdir()):
        if not sim_folder.is_dir():
            continue
        
        sim_name = sim_folder.name
        x_traj_file = traj_dir / f"xc_{sim_name}.csv"
        y_traj_file = traj_dir / f"yc_{sim_name}.csv"

        if not (x_traj_file.exists() and y_traj_file.exists()):
            print(f"Trajectory files for simulation {sim_name} not found, skipping.")
            continue

        x_traj = pd.read_csv(x_traj_file, header=None).values
        y_traj = pd.read_csv(y_traj_file, header=None).values
        num_particles.append(x_traj.shape[1])
        
        process_args.append((sim_folder, x_traj, y_traj))
    print('Max number of particles:', max(num_particles))
    print('Min number of particles:', min(num_particles))
    print('Mean number of particles:', np.mean(num_particles))

    # Process in parallel with intermediate saves
    all_results = []
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i, result in enumerate(executor.map(process_sequence, process_args)):
            all_results.append(result)
            
            # Save a sample reference image from the first valid result
            if i == 0 and len(result[1]) > 0:
                # Get coordinates from the first frame
                sim_folder = process_args[0][0]
                x_traj = process_args[0][1]
                y_traj = process_args[0][2]
                
                # Create sample images
                sample_test = io.imread(sorted(Path(sim_folder).glob('*.jpg'))[0])
                ref_image, aug_test = create_reference_and_augment_test(
                    x_traj[0],
                    y_traj[0],
                    sample_test,
                    particle_diameter=8
                )
                
                # Save both reference and augmented test images
                io.imsave(output_dir / 'sample_reference_image.png', ref_image)
                io.imsave(output_dir / 'sample_augmented_test.png', aug_test)
                io.imsave(output_dir / 'sample_original_test.png', sample_test)
            
            # Save intermediate results every 50 folders
            if (i + 1) % 50 == 0:
                save_intermediate_results(all_results, output_dir, i + 1)

    # Save final results
    save_intermediate_results(all_results, output_dir, 'final')

    # Create final histograms
    df = pd.DataFrame([(sim, snr_orig, snr_aug) 
                      for sim, values in all_results 
                      for _, snr_orig, snr_aug in values],
                     columns=['simulation', 'snr_original', 'snr_augmented'])
    
    # Plot original SNR histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['snr_original'].dropna(), bins=30, edgecolor='black')
    plt.title('Distribution of Original SNR Values')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir / 'snr_histogram_original.png')
    plt.close()
    
    # Plot augmented SNR histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['snr_augmented'].dropna(), bins=30, edgecolor='black')
    plt.title('Distribution of Augmented SNR Values')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir / 'snr_histogram_augmented.png')
    plt.close()

    # Create violin plot
    plt.figure(figsize=(10, 6))
    data_to_plot = [
        df['snr_original'].dropna(),
        df['snr_augmented'].dropna()
    ]
    
    violin_parts = plt.violinplot(data_to_plot, 
                                 showmeans=True, 
                                 showmedians=True)
    
    # Customize violin plot
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    violin_parts['cmeans'].set_color('black')
    violin_parts['cmedians'].set_color('blue')
    violin_parts['cbars'].set_color('black')
    violin_parts['cmins'].set_color('black')
    violin_parts['cmaxes'].set_color('black')
    
    plt.xticks([1, 2], ['Original SNR', 'Augmented SNR'])
    plt.ylabel('SNR (dB)')
    plt.title('Distribution of SNR Values')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add box with statistics
    stats_text = f'Statistics:\n' \
                 f'Original SNR: {df["snr_original"].mean():.2f} ± {df["snr_original"].std():.2f}\n' \
                 f'Augmented SNR: {df["snr_augmented"].mean():.2f} ± {df["snr_augmented"].std():.2f}'
    
    plt.text(0.95, 0.05, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'snr_violin_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Analysis complete! Results saved in:", output_dir)

if __name__ == "__main__":
    main()
