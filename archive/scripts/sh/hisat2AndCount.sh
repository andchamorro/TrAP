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
    echo "Usage: $0 -a <Annotation saf file> -x <HISAT2 index filename prefix (minus the trailing .X.ht2)> [-i <Single-end fastq file> | -1 <Paired-end fastq file 1> -2 <Paired-end fastq file 2>] [-j <N> =1 default] -o <Output directory>"
    exit 1
}

log_message() {
    echo "$1"
}

threads=1
hisat="hisat2"
hisatarg=""
featurecounts="featureCounts"
# Parse arguments
while getopts "a:x:i:1:2:j:o:f:y:h" opt; do
    case $opt in
        a) saffile=$OPTARG
           log_message "-a <Annotation saf file> = $saffile"
           ;;
        x) indexfile=$OPTARG
           log_message "-x <HISAT2 index filename prefix> = $indexfile"
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
        f) featurecounts=$OPTARG
           log_message "-f <featureCounts path> = $featurecounts"
           ;;
        y) case $OPTARG in
            truseq)
                hisatarg="--dta --rna-strandness RF --rg PL:ILLUMINA"
                ;;
            nextera)
                hisatarg="--dta --rna-strandness FR --rg PL:NEXTSEQ"
                ;;
            *)
                hisatarg=$OPTARG
                ;;
           esac
           ;;
        h) print_usage_and_exit
           ;;
        *) print_usage_and_exit
           ;;
    esac
done

# Check required arguments
if [ -z "$saffile" ] || [ -z "$indexfile" ] || [ -z "$outputdir" ]; then
    echo "Error: missing required argument(s)"
    print_usage_and_exit
fi

if [ -z "$inputfile" ] && ([ -z "$pairedfile1" ] || [ -z "$pairedfile2" ]); then
    echo "Error: missing input file(s)"
    print_usage_and_exit
fi
# Create output directory if it doesn't exist
mkdir -p "$outputdir"

# Step 1: Align reads to indexfile genome
echo "[1/6] Aligning reads to indexfile genome..."
echo "hisat version: $(eval $hisat --version)"

$hisat --threads "$threads" \
    "$hisatarg" --very-sensitive \
    -X 600 --no-mixed \
    -x "${indexfile}" -1 "$pairedfile1" -2 "$pairedfile2" \
    -S "$outputdir/aligned_reads.sam" \
    --summary-file "$outputdir/hisat2.out" --new-summary \
    --met-file "$outputdir/hisat2.met"

# Step 2: Convert SAM to BAM
echo "[2/6] Converting SAM to BAM..."
samtools view --threads "$threads" -Sb "$outputdir/aligned_reads.sam" > "$outputdir/aligned_reads.bam"

# Step 3: Sort BAM file
echo "[3/6] Sorting BAM file..."
samtools sort --threads "$threads" "$outputdir/aligned_reads.bam" -o "$outputdir/sorted_reads.bam"

# Step 4: Index BAM file
echo "[4/6] Indexing BAM file..."
samtools index --threads "$threads"  "$outputdir/sorted_reads.bam"

# Step 5: Count unique alignments
echo "[5/6] Counting unique alignments..."
$featurecounts -p -F "SAF" -M --fraction -a "$saffile" -o "$outputdir/counts.txt" "$outputdir/sorted_reads.bam"

trap : 0
echo "All steps completed successfully."
echo '
**********************************************
*** DONE Processing ...
*** You can use the file `$outfolder/counts.txt`
**********************************************
'
