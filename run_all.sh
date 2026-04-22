#!/bin/bash

mkdir -p logs outputs

declare -A DATASET_STYLES
DATASET_STYLES["common-pile/arxiv_papers_filtered"]="math table faq"
DATASET_STYLES["common-pile/data_provenance_initiative_filtered"]="math"
DATASET_STYLES["common-pile/libretexts_filtered"]="math diverse_qa_pairs"
DATASET_STYLES["common-pile/peS2o_filtered"]="math diverse_qa_pairs"
DATASET_STYLES["common-pile/oercommons_filtered"]="tutorial diverse_qa_pairs"
DATASET_STYLES["common-pile/stackexchange_filtered"]="tutorial discussion"
DATASET_STYLES["common-pile/stackv2_edu_filtered"]="tutorial"
DATASET_STYLES["common-pile/pubmed_filtered"]="table diverse_qa_pairs"
DATASET_STYLES["common-pile/uk_hansard_filtered"]="discussion diverse_qa_pairs"
DATASET_STYLES["common-pile/caselaw_access_project_filtered"]="discussion"
DATASET_STYLES["common-pile/cccc_filtered"]="discussion"
DATASET_STYLES["common-pile/doab_filtered"]="discussion faq"
DATASET_STYLES["common-pile/github_archive_filtered"]="discussion diverse_qa_pairs"
DATASET_STYLES["common-pile/news_filtered"]="discussion"
DATASET_STYLES["common-pile/foodista_filtered"]="discussion"
DATASET_STYLES["common-pile/pre_1929_books_filtered"]="discussion"
DATASET_STYLES["common-pile/pressbooks_filtered"]="discussion"
DATASET_STYLES["common-pile/project_gutenberg_filtered"]="discussion"
DATASET_STYLES["common-pile/python_enhancement_proposals_filtered"]="discussion"
DATASET_STYLES["common-pile/stackv2_html_filtered"]="discussion"
DATASET_STYLES["common-pile/ubuntu_irc_filtered"]="discussion"
DATASET_STYLES["common-pile/youtube_filtered"]="discussion diverse_qa_pairs"
DATASET_STYLES["common-pile/biodiversity_heritage_library_filtered"]="faq"
DATASET_STYLES["common-pile/library_of_congress_filtered"]="diverse_qa_pairs"
DATASET_STYLES["common-pile/regulations_filtered"]="diverse_qa_pairs"
DATASET_STYLES["common-pile/usgpo_filtered"]="diverse_qa_pairs"
DATASET_STYLES["common-pile/uspto_filtered"]="diverse_qa_pairs"
DATASET_STYLES["common-pile/public_domain_review_filtered"]="diverse_qa_pairs"
DATASET_STYLES["common-pile/wikimedia_filtered"]="diverse_qa_pairs"
DATASET_STYLES["common-pile/wikiteam_filtered"]="diverse_qa_pairs"

for dataset in "${!DATASET_STYLES[@]}"; do
    for style in ${DATASET_STYLES[$dataset]}; do
        export DATASET="$dataset"
        export STYLE="$style"
        export DATASET_SAFE="${dataset//\//__}"
        sbatch --job-name="synth-${DATASET_SAFE}-${style}" run_single.sh
        echo "Submitted: $dataset × $style"
    done
done

echo "All jobs submitted."
