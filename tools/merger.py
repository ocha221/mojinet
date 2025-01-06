from pathlib import Path
import shutil
from tqdm import tqdm
import logging
from collections import defaultdict
import multiprocessing as mp
import json
import os
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_dict(data, chunks):
    items = list(data.items())
    avg_size = len(items) // chunks
    remainder = len(items) % chunks

    result = []
    start = 0
    for i in range(chunks):
        chunk_size = avg_size + (1 if i < remainder else 0)
        chunk_items = items[start : start + chunk_size]
        result.append(dict(chunk_items))
        start += chunk_size

    return result


def safe_json_serialize(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        logger.error(f"Error serializing JSON: {e}")
        # * uh oh
        sanitized = {
            k: str(v)
            .replace('"', '\\"')
            .replace("\n", "\\n")  # * just escaping unsafe chars really
            for k, v in obj.items()
        }
        return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))


def process_character(args):
    character, grid_paths, final_dir, worker_id, dry_run = args
    stats = {
        "worker_id": worker_id,
        "character": character,
        "input_files": 0,
        "output_files": 0,
        "source_grids": len(grid_paths),
        "operations": [],  # * for debugging your dry runs
    }

    char_output_dir = Path(final_dir) / character
    if not dry_run:
        os.makedirs(char_output_dir, exist_ok=True)

    grid_info = {}
    all_files = []  # * List to store all files we find

    #* collecting

    for grid_path in grid_paths:
        char_dir = Path(grid_path) / character
        if char_dir.exists():
            files_in_grid = [
                entry.path
                for entry in os.scandir(char_dir)
                if entry.is_file() and entry.name.endswith(".png")
            ]
            all_files.extend(files_in_grid)

            grid_info[str(grid_path)] = {
                "file_count": len(files_in_grid),
                "path": str(grid_path),
                "start_idx": global_counter,
                "end_idx": global_counter + len(files_in_grid) - 1,
            }

            for src_file in files_in_grid:
                dst_file_str = str(char_output_dir / f"{global_counter}.png")

                if dry_run:
                    stats["operations"].append(
                        {
                            "source": src_file,
                            "destination": dst_file_str,
                            "index": global_counter,
                        }
                    )
                else:
                    shutil.copy2(src_file, dst_file_str)

                global_counter += 1
                stats["input_files"] += 1
                stats["output_files"] += 1

    stats_file = Path(final_dir) / f"worker_{worker_id}_stats.json"
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json_str = safe_json_serialize(stats)
            f.write(json_str + "\n")
    except Exception as e:
        logger.error(f"Error writing stats for worker {worker_id}: {e}")

    return character, grid_info


def fast_directory_scan(
    worker_dirs,
):  # * this builds a dict of *every* symbol -> grid folders its found in
    results = defaultdict(dict)

    for worker_dir in worker_dirs:

        for grid_entry in os.scandir(worker_dir):
            if not grid_entry.is_dir():
                continue

            for symbol_entry in os.scandir(grid_entry.path):
                if not symbol_entry.is_dir():
                    continue

                results[symbol_entry.name][grid_entry.path] = {"path": grid_entry.path}

    return results


def merge_worker_outputs(dry_run, output_dir, final_dir, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()

    logger.info(
        f"Starting {'dry run' if dry_run else 'merge'} with {num_workers} workers"
    )

    worker_dirs = list(Path(output_dir).glob("worker_*"))

    logger.info("Building initial index, hold...")
    start_time = time.time()

    symbol_to_grids = fast_directory_scan(worker_dirs)

    elapsed_time = time.time() - start_time
    logger.info(
        f"✨ Initial index built in {elapsed_time:.2f} seconds! Found {len(symbol_to_grids)} unique symbols"
    )

    if not dry_run:
        os.makedirs(final_dir, exist_ok=True)

    if os.path.exists(final_dir):  # * This clears old worker files, not the images
        for stats_file in Path(final_dir).glob("worker_*_stats.json"):
            try:
                os.remove(stats_file)
            except Exception as e:
                logger.error(f"Error removing old stats file {stats_file}: {e}")

    logger.info(
        f"Starting merge with {len(os.listdir(output_dir))} files in output_dir"
    )

    char_chunks = chunk_dict(symbol_to_grids, num_workers)

    process_args = []
    for worker_id, chunk in enumerate(char_chunks):
        for char, grids in chunk.items():
            process_args.append((char, grids, final_dir, worker_id, dry_run))
    if not dry_run:
        logger.info(f"Writing to disk, be patient...")
    logger.info(
        f"Processing {len(symbol_to_grids)} characters, {len(process_args)} total entries with {num_workers} workers. "
    )
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_character, process_args),
                total=len(process_args),
                desc="Processing characters",
            )
        )
    print("Finished processing! Gathering stats, please wait...")

    final_mapping = {char: grid_info for char, grid_info in results}

    total_stats = defaultdict(int)
    operations = []
    for stats_file in Path(final_dir).glob("worker_*_stats.json"):
        try:
            with open(stats_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip(): 
                        try:
                            stats = json.loads(line)
                            total_stats["input_files"] += stats["input_files"]
                            total_stats["output_files"] += stats["output_files"]
                            total_stats["characters"] += 1
                            if dry_run and "operations" in stats:
                                operations.extend(stats["operations"])
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Error parsing JSON line in {stats_file}: {e}"
                            )
                            continue
        except Exception as e:
            logger.error(f"Error reading stats file {stats_file}: {e}")
            continue

    unique_sources = {op["source"] for op in operations}
    logger.info(f"Total operations: {len(operations)}")
    logger.info(f"Unique source files: {len(unique_sources)}")
    logger.info("Sample of files to be processed:")
    for source in list(unique_sources)[:5]:
        logger.info(f"- {source}")

    logger.info("\nProcessing Summary:")
    logger.info(f"Total characters processed: {total_stats['characters']}")
    logger.info(f"Total input files: {total_stats['input_files']}")
    logger.info(f"Total output files: {total_stats['output_files']}")

    if dry_run:
        logger.info("\nPlanned operations (first 10 examples):")
        for op in operations[:10]:
            logger.info(
                f"Would copy {op['source']} -> {op['destination']} (file {op['index']})"
            )
        if len(operations) > 10:
            logger.info(f"... and {len(operations) - 10} more operations")

    return final_mapping


if __name__ == "__main__":
    OUTPUT_DIR = "point to the output of tiles_from_pairs.py, default etl_worker"
    FINAL_DIR = "your choice"

    # * Clear the final directory first to avoid accumulation.
    #! This will delete all created files in the final directory, be careful!
    #! It might take a moment! Its > 1 million items
    if os.path.exists(FINAL_DIR) and True:
        logger.info(f"Clearing {FINAL_DIR} before run")
        files = os.listdir(FINAL_DIR)
        for f in tqdm(files, desc="Clearing directory"):
            try:
                os.remove(os.path.join(FINAL_DIR, f))
            except Exception as e:
                logger.error(f"Failed to remove {f}: {e}")
                exit(1)
        logger.info("Directory cleared")

    # * Change dry_run to False to commit changes to disk
    # * Run with dry first to check out worker jsons, and the sample output.
    # * Dry_run = False will not print stats.
    mapping = merge_worker_outputs(
        dry_run=True, output_dir=OUTPUT_DIR, final_dir=FINAL_DIR
    )
    print(f"\n{len(mapping)} unique characters")
