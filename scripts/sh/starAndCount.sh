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
star="STAR"
# Parse arguments
while getopts "a:r:g:s:i:1:2:m:f:j:o:h" opt; do
    case $opt in
        a) saffile=$OPTARG
           log_message "-a <Annotation saf file> = $saffile"
           ;;
        r) reference=$OPTARG
           log_message "-r <Reference genome file> = $reference"
           ;;
        g) sjdbGTFfile=$OPTARG
           log_message "-g <sjdbGTF file> = $sjdbGTFfile"
           ;;
        s) stardir=$OPTARG
           log_message "-s <STAR reference directory> = $stardir"
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
        m) anchormultimapnmax=$OPTARG
           log_message "-m winAnchorMultimapNmax = $anchormultimapnmax"
           ;;
        f) filtermultimapnmax=$OPTARG
           log_message "-f outFilterMultimapNmax = $filtermultimapnmax"
           ;;
        j) threads="$OPTARG"
           log_message "-j <Concurrency level> = $threads"
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

echo "STAR version: $(eval $star --version)"
if [ ! -d "$stardir" ]; then
    echo "[0/6] star index not found. Generate..."
    $star --runMode genomeGenerate \
        --runThreadN "$threads" \
        --genomeDir "$stardir" \
        --genomeFastaFiles "$reference" \
        --genomeSAindexNbases 14 \
        --genomeChrBinNbits 18 \
        --genomeSAsparseD 3 \
        --limitGenomeGenerateRAM 17179869184 \
        --sjdbGTFfile "$sjdbGTFfile"
    echo "star index generate."
fi

# Step 1: Align reads to reference genome
echo "[1/6] Aligning reads to reference genome..."
$star --runThreadN "$threads" \
     --genomeDir "$stardir" \
     --readFilesIn "$pairedfile1" "$pairedfile2" \
     --readFilesCommand cat \
     --readNameSeparator space \
     --outSAMunmapped Within KeepPairs \
     --outSAMtype BAM Unsorted \
     --outStd Log \
     --winAnchorMultimapNmax "$anchormultimapnmax" \
     --outFilterMultimapNmax "$filtermultimapnmax" \
     --outFileNamePrefix "$outputdir/" > "$outputdir/star.out"

# Step 3: Sort BAM file
echo "[3/6] Sorting BAM file..."
samtools sort -@ "$threads" "$outputdir/Aligned.out.bam" -o "$outputdir/sorted.bam"

# Step 4: Index BAM file
echo "[4/6] Indexing BAM file..."
samtools index "$outputdir/sorted.bam"

# Step 5: Count unique alignments
echo "[5/6] Counting unique alignments..."
featureCounts -p -F "SAF" -M --fraction -a "$saffile" -o "$outputdir/counts.txt" "$outputdir/sorted.bam"

trap : 0
echo "All steps completed successfully."
echo '
**********************************************
*** DONE Processing ...
*** You can use the file `$outfolder/counts.txt`
**********************************************
'
