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
bwa=$(which bwa)
# Parse arguments
while getopts "a:r:i:1:2:j:o:h" opt; do
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


echo "bwa $(eval $bwa 2>&1 | grep -m 1 -i Version)"
if [ ! -f "${reference}.bwt" ]; then
    echo "[0/6] star index not found. Creating index..."
    $bwa index "$reference"
    echo "bwa index created."
fi

# Step 1: Align reads to reference genome
echo "[1/4] Aligning reads to reference genome..."
basepairedfile=$(basename "$pairedfile1")
$bwa aln -t "$threads" -N -n 3 -k 3 -i 20 -R 10000000 "$reference" "$pairedfile1" > "$outputdir/${basepairedfile%.*}_1.aln.sai" &
$bwa aln -t "$threads" -N -n 3 -k 3 -i 20 -R 10000000 "$reference" "$pairedfile2" > "$outputdir/${basepairedfile%.*}_2.aln.sai" &
wait

# Step 2: Sampe to sorted BAM
echo "[2/4] Sampe to sorted BAM..."
$bwa sampe -n 10000000 -N 10000000 "$reference" \
    "$outputdir/${basepairedfile%.*}_1.aln.sai" \
    "$outputdir/${basepairedfile%.*}_2.aln.sai" \
    "$pairedfile1" "$pairedfile2" > "$outputdir/aligned_reads.sam"

# Step 2: Convert SAM to BAM
echo "[2/6] Converting SAM to BAM..."
samtools view --threads "$threads" -Sb "$outputdir/aligned_reads.sam" > "$outputdir/aligned_reads.bam"

# Step 3: Sort BAM file
echo "[3/6] Sorting BAM file..."
samtools sort --threads "$threads" "$outputdir/aligned_reads.bam" -o "$outputdir/sorted_reads.bam"

# Step 3: Index BAM file
echo "[3/4] Indexing BAM file..."
samtools index "$outputdir/sorted_reads.bam"

# Step 5: Count unique alignments
# echo "[4/4] Counting unique alignments..."
# featureCounts -F "SAF" -M --fraction -a "$saffile" -o "$outputdir/counts.txt" "$outputdir/sorted_reads.bam"

trap : 0
echo "All steps completed successfully."
echo '
**********************************************
*** DONE Processing ...
*** You can use the file `$outfolder/counts.txt`
**********************************************
'
