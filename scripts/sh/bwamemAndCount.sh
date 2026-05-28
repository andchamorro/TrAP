#!/bin/bash

abort()
{
    echo >&2 '
***************
*** ABORTED ***
***************
'
    echo "An error occurred. Exiting..." >&2
    exit 1
}

trap 'abort' 0

print_usage_and_exit() {
    echo "Usage: $0 -a <Annotation saf file> -r <Reference genome file> [-i <Single-end fastq file> | -1 <Paired-end fastq file 1> -2 <Paired-end fastq file 2>] [-j <N> =1 default] -o <Output directory>"
    exit 1
}

log_message() {
    echo "$1"
}

threads=1
bwa="bwa"
featureCounts="featureCounts"
# Parse arguments
while getopts "a:r:i:1:2:f:j:o:h" opt; do
    case $opt in
        a) saffile=$OPTARG
           log_message "-a <Annotation saf file> = $saffile"
           ;;
        r) reference=$OPTARG
           log_message "-r <Reference genome file> = $reference"
           ;;
        i) inputfile=$OPTARG
           log_message "-i <Single-end fastq file> = $inputfile"
           ;;
        1) pairedfile1=$OPTARG
           log_message "-1 <Paired-end fastq file 1> = $pairedfile1"
           ;;
        2) pairedfile2=$OPTARG
           log_message "-2 <Paired-end fastq file 2> = $pairedfile2"
           ;;
        f) featureCounts=$OPTARG
           log_message "-f <featureCounts Path> = $featureCounts"
           ;;
        j) threads="$OPTARG"
           echo "-j <Concurrency level> = $threads"
           ;;
        o) outputdir=$OPTARG
           log_message "-o <Output directory> = $outputdir"
           ;;
        h) print_usage_and_exit
           ;;
        *) print_usage_and_exit
           ;;
    esac
done

# Check required arguments
if [ -z "$saffile" ] || [ -z "$reference" ] || [ -z "$outputdir" ]; then
    echo "Error: missing required argument(s)"
    print_usage_and_exit
fi

if [ -z "$inputfile" ] && ([ -z "$pairedfile1" ] || [ -z "$pairedfile2" ]); then
    echo "Error: missing input file(s)"
    print_usage_and_exit
fi
# Create output directory if it doesn't exist
mkdir -p "$outputdir"

# Step 1: Align reads to reference genome
echo "[1/6] Aligning reads to reference genome..."
echo "bwa $(eval $bwa 2>&1 | grep -m 1 -i Version)"
if [ ! -f "${reference}.bwt" ]; then
    echo "bwa index not found. Creating index..."
    $bwa index "$reference"
    echo "bwa index created."
fi

# Check if paired-end files are provided
if [[ -n "$pairedfile1" && -n "$pairedfile2" ]]; then
    echo "Running BWA-MEM for paired-end reads..."
    bwa mem -t "$threads" "$reference" "$pairedfile1" "$pairedfile2" > "$outputdir/aligned_reads.sam"
else
    echo "Running BWA-MEM for single-end reads..."
    bwa mem -t "$threads" "$reference" "$inputfile" > "$outputdir/aligned_reads.sam"
fi

# Step 2: Convert SAM to BAM
echo "[2/6] Converting SAM to BAM..."
samtools view -Sb "$outputdir/aligned_reads.sam" > "$outputdir/aligned_reads.bam"

# Step 3: Sort BAM file
echo "[3/6] Sorting BAM file..."
samtools sort --threads "$threads" "$outputdir/aligned_reads.bam" -o "$outputdir/sorted_reads.bam"

# Step 4: Index BAM file
echo "[4/6] Indexing BAM file..."
samtools index --threads "$threads"  "$outputdir/sorted_reads.bam"

# Step 5: Count unique alignments
echo "[5/6] Counting unique alignments..."
$featureCounts -F "SAF" -M --fraction -a "$saffile" -o "$outputdir/counts.txt" "$outputdir/sorted_reads.bam"

trap : 0
echo "All steps completed successfully."
echo '
**********************************************
*** DONE Processing ...
*** You can use the file `$outfolder/counts.txt`
**********************************************
'
