#!/bin/bash
#SBATCH --job-name=genome_processing
#SBATCH --output=genome_processing_%j.out
#SBATCH --error=genome_processing_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=48:00:00

# Load necessary modules
module load GCC/13.2.0 STAR/2.7.11b
module load SAMtools/1.21
module load BEDTools/2.31.1

# Set variables
REF_PATH="/path/to/reference/genome"
REF_DIR="/path/to/reference/directory"
REPBASE_PATH="/path/to/reference/repbase"
GENOME_NAME=$(basename "$REF_PATH")
GENOME_NAME="${GENOME_NAME%.*}"

# Create directories
mkdir -p art star

# Generate reads with ART
art_illumina -sam -na -i $REF_PATH -p -l 150 -f 2 -m 500 -s 10 -ss MSv3 -o art/$GENOME_NAME.2x_R

# Run STAR alignment
STAR --runThreadN 32 \
    --genomeDir $REF_DIR \
    --readFilesIn art/${GENOME_NAME}.2x_R1.fq art/${GENOME_NAME}.2x_R2.fq \
    --readFilesCommand cat \
    --readNameSeparator space \
    --outSAMunmapped Within KeepPairs \
    --outSAMtype BAM Unsorted \
    --outStd Log \
    --winAnchorMultimapNmax 100 \
    --outFilterMultimapNmax 100 \
    --outFileNamePrefix star/

# Sort BAM files with samtools
samtools sort -@48 star/Aligned.out.bam -o star/assigned_sorted.bam > star/samtools_sort.out &
samtools sort -@48 -n -o star/assigned_sorted_byn.bam star/assigned_sorted.bam

# Pair BAM files with bedtools
GFF_DIR="gffs"
# Call the Python script to process the GFF3 file
python divide_gff.py $REPBASE_PATH $GFF_DIR

# Generate BAM files using bedtools
BAM_DIR="BAM_LINEs"
FASTQ_DIR="FASTQ_LINEs"

mkdir -p $BAM_DIR
mkdir -p $FASTQ_DIR

for gff_file in $GFF_DIR/*.gff; do
    # Create a bam for each target
    target=$(basename $gff_file .gff)
    bam_file_path="$BAM_DIR/${GENOME_NAME}.2x.paired.p14_rm.${target}.bam"
    err_file_path="$BAM_DIR/${GENOME_NAME}.2x.paired.p14_rm.${target}.err"
    bedtools pairtobed -type either -abam star/assigned_sorted_byn.bam -b $gff_file > $bam_file_path 2>> $err_file_path
    echo "Created BAM file for target '${target}': ${bam_file_path}"

    # Convert BAM to FASTQ and modify read IDs
    fastq_r1_path="$FASTQ_DIR/${GENOME_NAME}.2x.paired.p14_rm.${target}_R1"
    fastq_r2_path="$FASTQ_DIR/${GENOME_NAME}.2x.paired.p14_rm.${target}_R2"
    samtools fastq -f 1 -F 268 -1 "${fastq_r1_path}.nsec" -2 "${fastq_r2_path}.nsec" $bam_file_path
    samtools fastq -f 256 -F 12 -1 "${fastq_r1_path}.sec" -2 "${fastq_r2_path}.nsec" $bam_file_path
    cat "${fastq_r1_path}.nsec" "${fastq_r1_path}.sec" > "${fastq_r1_path}.fq"
    cat "${fastq_r2_path}.nsec" "${fastq_r2_path}.sec" > "${fastq_r2_path}.fq"
    awk -v target="$target" '{if(NR%4==1) {print $0 "|" target} else {print $0}}' "${fastq_r1_path}.fq" > "${fastq_r1_path}.tmp" && mv "${fastq_r1_path}.tmp" "${fastq_r1_path}.fq"
    awk -v target="$target" '{if(NR%4==1) {print $0 "|" target} else {print $0}}' "${fastq_r2_path}.fq" > "${fastq_r2_path}.tmp" && mv "${fastq_r2_path}.tmp" "${fastq_r2_path}.fq"
    echo "Created FASTQ files for target '${target}': ${fastq_r1_path}, ${fastq_r2_path}"
done

# Concatenate all R1 and R2 FASTQ files
cat $FASTQ_DIR/*_R1.fq > "${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_R1.fq"
cat $FASTQ_DIR/*_R2.fq > "${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_R2.fq"

echo "Concatenation complete: ${GENOME_NAME}.2x.paired.p14_rm.LINE1_R1.fq and ${GENOME_NAME}.2x.paired.p14_rm.LINE1_R2.fq"

# bedtools pairtobed -type either -abam ${GENOME_NAME}.2x.paired/assigned_sorted_byn.bam -b $REPBASE_PATH > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1.bam 2>> ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1.err
bedtools pairtobed -type neither -abam ${GENOME_NAME}.2x.paired/assigned_sorted_byn.bam -b $REPBASE_PATH > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1.bam 2>> ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1.err

# Extract FASTQ files with samtools
# samtools fastq -f 1 -F 268 -1 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_nsec_R1.fq -2 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_nsec_R2.fq -s ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_nsec_R.fq ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1.bam
# samtools fastq -f 256 -F 12 -1 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_sec_R1.fq -2 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_sec_R2.fq -s ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_sec_R.fq ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1.bam

samtools fastq -f 1 -F 268 -1 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_nsec_R1.fq -2 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_nsec_R2.fq -s ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_nsec_R.fq ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1.bam
samtools fastq -f 256 -F 12 -1 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_sec_R1.fq -2 ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_sec_R2.fq -s ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_sec_R.fq ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1.bam

# Concatenate FASTQ files
# cat ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_*_R.fq > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_R.fq
# cat ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_*_R1.fq > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_R1.fq
# cat ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_*_R2.fq > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_R2.fq
cat ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_*_R.fq > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_R.fq
cat ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_*_R1.fq > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_R1.fq
cat ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_*_R2.fq > ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_R2.fq

# Remove intermediate FASTQ files
# rm -rf ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_*_R1.fq
# rm -rf ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_*_R2.fq
# rm -rf ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_unmapxunmap_*_R.fq
rm -rf ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_*_R1.fq
rm -rf ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_*_R2.fq
rm -rf ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_unmapxunmap_*_R.fq

# Run subsampling script
# python make_subsample_paired.py ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.LINE1_R ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14_rm.NOTLINE1_R ${GENOME_NAME}.2x.paired/${GENOME_NAME}.2x.paired.p14.subsampling_163840 81920 1 4096