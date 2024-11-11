import os
import sys
import json
import time
import gzip
from habanero import Crossref
import polars as pl
import logging

# -------------------------------
# Configuration
# -------------------------------

# Replace with your actual email
CROSSREF_MAILTO = "youremail@example.com"

# Directories and file paths
INPUT_JSONL = r'F:\repos\ml_embs\app\processed_dois.jsonl'  # Path to your input JSONL file with 'doi' and 'filename' fields
METADATA_DIR = 'doi_metadata'                               # Directory to save batch metadata files
PROCESSED_DOIS_FILE = 'processed_dois.txt'                  # File to keep track of processed DOIs

# Settings
BATCH_SIZE = 100          # Number of DOIs per batch (CrossRef limit)
MAX_RETRIES = 5          # Maximum number of retries for failed requests

# -------------------------------
# Logging Setup
# -------------------------------

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# Functions
# -------------------------------

def load_processed_dois(processed_dois_file):
    """
    Load the set of processed DOIs from a file.
    """
    if os.path.exists(processed_dois_file):
        with open(processed_dois_file, 'r', encoding='utf-8') as f:
            processed_dois = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(processed_dois)} processed DOIs from {processed_dois_file}")
    else:
        processed_dois = set()
    return processed_dois

def save_processed_dois(processed_dois, processed_dois_file):
    """
    Save the set of processed DOIs to a file.
    """
    with open(processed_dois_file, 'w', encoding='utf-8') as f:
        for doi in processed_dois:
            f.write(f"{doi}\n")
    logger.info(f"Saved {len(processed_dois)} processed DOIs to {processed_dois_file}")

def fetch_metadata_batch(doi_batch):
    """
    Fetch metadata for a batch of DOIs using CrossRef API.
    Returns a list of (doi, metadata) tuples.
    """
    cr = Crossref(mailto=CROSSREF_MAILTO)
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # Fetch metadata for the batch of DOIs
            response = cr.works(ids=doi_batch)
            results = []
            if isinstance(response, list):
                # Multiple DOIs response
                for res in response:
                    if 'message' in res:
                        item = res['message']
                        doi = item.get('DOI')
                        if doi:
                            results.append((doi.lower(), item))
            elif isinstance(response, dict):
                # Single DOI response
                item = response.get('message')
                doi = item.get('DOI')
                if doi:
                    results.append((doi.lower(), item))
            else:
                logger.error(f"Unexpected response type: {type(response)}")
            return results
        except Exception as e:
            logger.error(f"Error fetching metadata for batch: {e}")
            attempt += 1
            if attempt < MAX_RETRIES:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1} of {MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch metadata after {MAX_RETRIES} attempts.")
                break
    return []

def save_batch_compressed(results, batch_num, metadata_dir, processed_dois):
    """
    Save metadata from a batch of results into a compressed JSON Lines file and update the set of processed DOIs.
    """
    if results:
        batch_filename = f"batch_{batch_num}.jsonl.gz"
        batch_filepath = os.path.join(metadata_dir, batch_filename)
        try:
            with gzip.open(batch_filepath, 'wt', encoding='utf-8') as f:
                for doi, metadata in results:
                    json_line = json.dumps(metadata, ensure_ascii=False)
                    f.write(json_line + '\n')
                    processed_dois.add(doi)
            logger.info(f"Saved batch {batch_num} metadata to {batch_filepath}")
        except Exception as e:
            logger.error(f"Error saving batch {batch_num} metadata to {batch_filepath}: {e}")
    else:
        logger.warning(f"No metadata to save for batch {batch_num}")

# -------------------------------
# Main Execution
# -------------------------------

def main():
    # Step 1: Read DOIs from the JSONL file
    logger.info(f"Reading DOIs from {INPUT_JSONL}")
    try:
        df_dois = pl.read_ndjson(INPUT_JSONL)
    except Exception as e:
        logger.error(f"Error reading JSONL file {INPUT_JSONL}: {e}")
        return

    # Ensure 'doi' field exists
    if 'doi' not in df_dois.columns:
        logger.error("'doi' field not found in the JSONL file.")
        return

    df_dois = df_dois.with_columns(
        pl.col('doi').str.to_lowercase()
    )

    doi_list = df_dois['doi'].to_list()

    if not doi_list:
        logger.error("No DOIs found in the JSONL file. Exiting.")
        return

    # Load the set of processed DOIs
    processed_dois = load_processed_dois(PROCESSED_DOIS_FILE)

    # Filter out already processed DOIs
    remaining_dois = [doi for doi in doi_list if doi not in processed_dois]

    if not remaining_dois:
        logger.info("All DOIs have been processed. Exiting.")
        return

    # Ensure the directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Split the remaining DOIs into batches
    total_dois = len(remaining_dois)
    num_batches = (total_dois + BATCH_SIZE - 1) // BATCH_SIZE
    doi_batches = [remaining_dois[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(num_batches)]

    for batch_num, doi_batch in enumerate(doi_batches, start=1):
        logger.info(f"Processing batch {batch_num}/{num_batches} with {len(doi_batch)} DOIs")
        results = fetch_metadata_batch(doi_batch)
        if results:
            save_batch_compressed(results, batch_num, METADATA_DIR, processed_dois)
        else:
            logger.warning(f"No metadata fetched for batch {batch_num}")

        # Save the current state after each batch
        save_processed_dois(processed_dois, PROCESSED_DOIS_FILE)

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
