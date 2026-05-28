import pandas as pd
import numpy as np
import argparse

def main(file_path, power):
    # Load the TSV file
    df = pd.read_csv(file_path, sep='\t')

    # Compute mean of NumReads
    mean_reads = df['NumReads'].mean()

    # Define standard deviation proportional to power
    std_dev = mean_reads * (1.5*power / 100.0)

    # Add Gaussian noise centered around the original mean
    np.random.seed(42)  # for reproducibility
    noise = np.random.normal(loc=mean_reads, scale=std_dev, size=len(df))
    df['NumReads'] = df['NumReads'] + noise - mean_reads  # keep mean unchanged
    df['NumReads'] = df['NumReads'].clip(lower=0)  # Ensure no negative values

    # Recalculate TPM
    norm_factor = (df['NumReads'] / df['EffectiveLength']).sum()
    df['TPM'] = (df['NumReads'] / df['EffectiveLength']) / norm_factor * 1e6

    # Save the updated DataFrame back to the same file
    df.to_csv(file_path, sep='\t', index=False)
    print(f"Updated file saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise to NumReads and recalculate TPM.")
    parser.add_argument("file_path", help="Path to the input TSV file")
    parser.add_argument("power", type=float, help="Power level to scale noise standard deviation")
    args = parser.parse_args()
    main(args.file_path, args.power)
