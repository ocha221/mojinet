from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict
import multiprocessing as mp
import json
import os
import time
import fcntl


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


def move_to(src, dst, buffer_size=1024 * 1024):
    try:
        with open(src, "rb") as fsrc:
            with open(dst, "wb") as fdst:

                src_fd = fsrc.fileno()
                dst_fd = fdst.fileno()

                try:
                    os.sendfile(dst_fd, src_fd, 0, os.path.getsize(src))
                    return True
                except (AttributeError, OSError):
                    while True:
                        buf = os.read(src_fd, buffer_size)
                        if not buf:
                            break
                        os.write(dst_fd, buf)
                    return True
    except OSError as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False


def batch_process_files(files, dst_dir, start_idx):
    results = []
    current_idx = start_idx

    for src_file in files:
        dst_file = os.path.join(dst_dir, f"{current_idx}.png")
        if move_to(src_file, dst_file):
            results.append((current_idx, src_file, dst_file))
        current_idx += 1

    return results


def process_character(args):
    character, grid_paths, final_dir, worker_id, dry_run = args
    stats = {
        "worker_id": worker_id,
        "character": character,
        "input_files": 0,
        "output_files": 0,
        "source_grids": len(grid_paths),
        "operations": [],
    }

    char_output_dir = os.path.join(final_dir, character)
    if not dry_run:
        os.makedirs(char_output_dir, exist_ok=True)

    grid_info = {}
    all_files = []

    for grid_path in grid_paths:
        char_dir = os.path.join(grid_path, character)
        if os.path.exists(char_dir):
            try:
                with os.scandir(char_dir) as scanner:
                    files_in_grid = [
                        entry.path
                        for entry in scanner
                        if entry.name.endswith(".png") and entry.is_file()
                    ]
                    all_files.extend(files_in_grid)

                    file_count = len(files_in_grid)
                    grid_info[str(grid_path)] = {
                        "file_count": file_count,
                        "path": str(grid_path),
                        "start_idx": stats["input_files"],
                        "end_idx": stats["input_files"] + file_count - 1,
                    }
                    stats["input_files"] += file_count
            except OSError as e:
                logger.error(f"Error scanning directory {char_dir}: {e}")
                continue

    if not dry_run and all_files:  # * actually process files
        batch_size = 1000
        file_counter = 0
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i : i + batch_size]
            results = batch_process_files(batch, char_output_dir, file_counter)
            stats["output_files"] += len(results)
            file_counter += len(results)

    if dry_run:
        stats["operations"].extend(
            [
                {"source": src, "destination": f"{char_output_dir}/{i}.png", "index": i}
                for i, src in enumerate(all_files)
            ]
        )

    # ? these dont matter, dont worry if you dont see any stats on success. failures should show though
    if not dry_run:
        stats_file = os.path.join(final_dir, f"worker_{worker_id}_stats.json")
        try:
            stats_json = json.dumps(stats, ensure_ascii=False, separators=(",", ":"))

            os.makedirs(os.path.dirname(stats_file), exist_ok=True)

            # ? you might also not need locks here but unlike locking writing images this doesnt really impact perfomance
            with open(stats_file, "w", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(stats_json + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.error(f"Error writing stats for worker {worker_id}: {e}")

    return character, grid_info


def merge_worker_outputs(
    dry_run, output_dir, final_dir, num_workers=None, batch_size=1000
):
    try:
        final_dir = os.path.abspath(final_dir)
        os.makedirs(final_dir, exist_ok=True)

        test_file = os.path.join(final_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except OSError as e:
        logger.error(f"Cannot write to final directory {final_dir}: {e}")
        raise

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)

    logger.info(
        f"Starting {'dry run' if dry_run else 'merge'} with {num_workers} workers"
    )

    worker_dirs = [
        d.path
        for d in os.scandir(output_dir)
        if d.name.startswith("worker_") and d.is_dir()
    ]

    logger.info("Building initial index, hold...")
    start_time = time.time()

    symbol_to_grids = fast_directory_scan(worker_dirs)

    elapsed_time = time.time() - start_time
    logger.info(
        f"✨ Initial index built in {elapsed_time:.2f} seconds! Found {len(symbol_to_grids)} unique symbols"
    )

    # * Always make sure final_dir exists
    os.makedirs(final_dir, exist_ok=True)

    # * clear
    if not dry_run and os.path.exists(final_dir):
        for entry in os.scandir(final_dir):
            if entry.name.endswith("_stats.json"):
                try:
                    os.remove(entry.path)
                except OSError as e:
                    logger.error(f"Error removing old stats file {entry.path}: {e}")

    char_chunks = chunk_dict(symbol_to_grids, num_workers)
    process_args = [
        (char, grids, final_dir, worker_id, dry_run)
        for worker_id, chunk in enumerate(char_chunks)
        for char, grids in chunk.items()
    ]

    if not dry_run:
        logger.info("Writing to disk, be patient...")
    logger.info(
        f"Processing {len(symbol_to_grids)} characters, {len(process_args)} total entries with {num_workers} workers."
    )

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_character, process_args),
                total=len(process_args),
                desc="Processing characters",
            )
        )
    if not dry_run:
        logger.info("Finished processing! Gathering stats...")

        final_mapping = {char: grid_info for char, grid_info in results if grid_info}

        total_stats = defaultdict(int)

        for stats_file in os.scandir(final_dir):
            if not stats_file.name.endswith("_stats.json"):
                continue
            try:
                with open(stats_file.path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            stats = json.loads(line)
                            total_stats["input_files"] += stats["input_files"]
                            total_stats["output_files"] += stats["output_files"]
                            total_stats["characters"] += 1
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Error parsing JSON in {stats_file.path}: {e}"
                            )
            except OSError as e:
                logger.error(f"Error reading stats file {stats_file.path}: {e}")

        logger.info("\nProcessing Summary:")
        logger.info(f"Total characters processed: {total_stats['characters']}")
        logger.info(f"Total input files: {total_stats['input_files']}")
        logger.info(f"Total output files: {total_stats['output_files']}")
    else:
        final_mapping = {char: grid_info for char, grid_info in results if grid_info}

        logger.info("\nDry Run Summary:")
        logger.info(f"Total characters to process: {len(final_mapping)}")
        total_files = sum(
            sum(grid["file_count"] for grid in char_info.values())
            for char_info in final_mapping.values()
        )
        logger.info(f"Total files to process: {total_files}")

    return final_mapping


if __name__ == "__main__":
    OUTPUT_DIR = "point to the output of tiles_from_pairs.py, default etl_worker"	 
    FINAL_DIR = "your choice"

    # * Clear the final directory first to avoid accumulation.
    #! This will delete all created files in the final directory, be careful!
    #! It might take a moment! Its > 1 million items
    if os.path.exists(FINAL_DIR) and False:
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
    start_time = time.time()
    mapping = merge_worker_outputs(
        dry_run=True, output_dir=OUTPUT_DIR, final_dir=FINAL_DIR
    )
    elapsed_time = time.time() - start_time
    print(f"\n{len(mapping)} unique characters")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
