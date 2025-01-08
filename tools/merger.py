from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict
import multiprocessing as mp
import json
import os
import time
import fcntl
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(message, style="bold blue"):
    console.print(f"ℹ️  {message}", style=style)

def log_error(message):
    console.print(f"❌ {message}", style="bold red")


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

    return character, grid_info, stats["input_files"], stats["output_files"]


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
        log_error(f"Cannot write to final directory {final_dir}: {e}")
        raise

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)

    log_info(f"Starting {'🔍 dry run' if dry_run else '🔄 merge'} with {num_workers} workers")

    worker_dirs = [
        d.path
        for d in os.scandir(output_dir)
        if d.name.startswith("worker_") and d.is_dir()
    ]

    with console.status("[bold green]Building initial index...") as status:
        start_time = time.time()
        symbol_to_grids = fast_directory_scan(worker_dirs)
        elapsed_time = time.time() - start_time
        log_info(f"✨ Initial index built in {elapsed_time:.2f} seconds!")
        log_info(f"Found {len(symbol_to_grids)} unique symbols")

    os.makedirs(final_dir, exist_ok=True)

    if os.path.exists(final_dir):
        with Progress() as progress:
            task = progress.add_task("🧹 Cleaning old stats files...", total=len(list(Path(final_dir).glob("worker_*_stats.json"))))
            for stats_file in Path(final_dir).glob("worker_*_stats.json"):
                try:
                    os.remove(stats_file)
                    progress.advance(task)
                except Exception as e:
                    log_error(f"Error removing old stats file {stats_file}: {e}")

    char_chunks = chunk_dict(symbol_to_grids, num_workers)
    process_args = [
        (char, grids, final_dir, worker_id, dry_run)
        for worker_id, chunk in enumerate(char_chunks)
        for char, grids in chunk.items()
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Processing characters...", total=len(process_args))
        with mp.Pool(num_workers) as pool:
            results = []
            for result in pool.imap_unordered(process_character, process_args):
                results.append(result)
                progress.advance(task)

    console.print("\n[bold green]✅ Processing complete![/]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    final_mapping = {char: grid_info for char, grid_info, _, _ in results if grid_info}    
    total_stats = defaultdict(int)
    operations = []
    
    with Progress() as progress:
        stats_files = list(Path(final_dir).glob("worker_*_stats.json"))
        task = progress.add_task("📊 Gathering stats...", total=len(stats_files))
        
        for stats_file in stats_files:
            try:
                with open(stats_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                stats = json.loads(line)
                                total_stats["input_files"] += stats["input_files"]
                                total_stats["output_files"] += stats["output_files"]
                                if dry_run and "operations" in stats:
                                    operations.extend(stats["operations"])
                            except json.JSONDecodeError as e:
                                log_error(f"Error parsing JSON line in {stats_file}: {e}")
                                continue
            except Exception as e:
                log_error(f"Error reading stats file {stats_file}: {e}")
            progress.advance(task)

    total_stats["characters"] = len(final_mapping)

    table.add_row("Characters Processed", str(total_stats['characters']))
    input_files = sum(stats[2] for stats in results)
    output_files = sum(stats[3] for stats in results)
    table.add_row("Input Files", str(input_files))
    table.add_row("Output Files", str(output_files))
    
    console.print("\n[bold]Processing Summary:[/]")
    console.print(table)

    if dry_run and operations:
        console.print("\n[bold]Planned Operations (First 10):[/]")
        for op in operations[:10]:
            source = Text(op['source'], style="blue")
            dest = Text(op['destination'], style="green")
            console.print(f"📄 {source} ➡️  {dest}")
        
        if len(operations) > 10:
            console.print(f"\n... and {len(operations) - 10} more operations")

    return final_mapping

if __name__ == "__main__":
    OUTPUT_DIR = "/Users/chai/Downloads/ETL 仮名・漢字 dataset/temp_workers"
    FINAL_DIR = "/Users/chai/Downloads/ETL 仮名・漢字 dataset/single iterator, old merger"

    console.print(Panel.fit(
        f"Label shards in: {OUTPUT_DIR}\nFinal Directory: {FINAL_DIR}",
        title="Configuration",
        border_style="blue"
    ))

    start_time = time.time()
    mapping = merge_worker_outputs(
        dry_run=False, output_dir=OUTPUT_DIR, final_dir=FINAL_DIR
    )
    elapsed_time = time.time() - start_time
    console.print(f"\n[bold green]✨ Processed {len(mapping)} unique characters[/]")
    console.print(f"[bold]Total execution time:[/] {elapsed_time:.2f} seconds")
