import cv2
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

console = Console()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def log_info(message, style="bold blue"):
    console.print(f"ℹ️  {message}", style=style)

def log_error(message):
    console.print(f"❌ {message}", style="bold red")

def log_warning(message):
    console.print(f"⚠️  {message}", style="bold yellow")

ETL_IMAGE_SIZES = {
    "ETL1": (64, 63),
    "ETL2": (60, 60),
    "ETL3": (72, 76),
    "ETL4": (72, 76),
    "ETL5": (72, 76),
    "ETL6": (64, 63),
    "ETL7": (64, 63),
    "ETL8B": (64, 63),
    "ETL8G": (128, 127),
    "ETL9B": (64, 63),
    "ETL9G": (128, 127),
}


def read_labels(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace("\n", "")
            return list(content)
    except UnicodeDecodeError:
        pass

    raise ValueError(f"Failed to read labels from {txt_path}")


def process_grid(args):
    image_path, txt_path, output_dir, worker_id = args

    try:
        etl_type = None
        for et in ETL_IMAGE_SIZES:
            if et in str(image_path):
                etl_type = et
                break

        if not etl_type:
            return {
                "status": "error",
                "error": f"Unknown ETL type for {image_path}",
                "file": str(image_path),
            }

        base_name = Path(image_path).stem
        worker_output = Path(output_dir) / f"worker_{worker_id}" / base_name
        worker_output.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "status": "error",
                "error": f"Failed to load image {image_path}",
                "file": str(image_path),
            }

        labels = read_labels(txt_path)
        cell_width, cell_height = ETL_IMAGE_SIZES[etl_type]
        rows = 40
        cols = 50

        stats = {
            "processed": 0,
            "skipped": 0,
            "empty_boxes": 0,
            "labels": {},
            "grid_file": str(image_path),
        }
        label_counters = {}
        current_pos = 0
        for row in range(rows):
            for col in range(cols):
                if current_pos >= len(labels):
                    break

                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height

                cell = img[y1:y2, x1:x2]
                label = labels[current_pos]

                if label == "\x00":
                    current_pos += 1
                    stats["empty_boxes"] += 1
                    continue

                if label not in label_counters:
                    label_counters[label] = 0

                counter_str = f"{label_counters[label]:05d}"

                label_dir = worker_output / label
                label_dir.mkdir(exist_ok=True)

                output_filename = f"{label}_{counter_str}.png"
                output_path = label_dir / output_filename

                if cv2.imwrite(str(output_path), cell):
                    stats["processed"] += 1
                    stats["labels"][label] = stats["labels"].get(label, 0) + 1
                    label_counters[label] += 1
                else:
                    stats["skipped"] += 1

                current_pos += 1

        return {"status": "success", "stats": stats, "label_counts": label_counters}

    except Exception as e:
        return {"status": "error", "error": str(e), "file": str(image_path)}


def main():
    parser = argparse.ArgumentParser(description="Slice ETL dataset grids into tiles")
    parser.add_argument("input", help="Base directory containing ETL dataset folders")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--temp", help="Temporary directory for worker outputs", default="temp_workers")
    args = parser.parse_args()

    base_dir = Path(args.input)
    if not base_dir.exists():
        log_error("Input directory not found!")
        return

    temp_dir = Path(args.temp)
    temp_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        ", ".join(ETL_IMAGE_SIZES.keys()),
        title="Available ETL Types",
        border_style="blue"
    ))

    user_input = input("Enter ETL type to process (or 'all' for all types): ").strip().upper()
    start_time = datetime.now()

    etl_types = list(ETL_IMAGE_SIZES.keys()) if user_input.lower() == 'all' else [user_input]
    
    pairs = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn()
    ) as progress:
        scan_task = progress.add_task("🔍 Scanning for PNG/TXT pairs...", total=len(etl_types))
        
        for etl_type in etl_types:
            if etl_type not in ETL_IMAGE_SIZES:
                log_warning(f"Unknown ETL type: {etl_type}, skipping...")
                progress.advance(scan_task)
                continue

            etl_dirs = list(base_dir.glob(f"*{etl_type}*"))
            if not etl_dirs:
                log_warning(f"No directories found for {etl_type}")
                progress.advance(scan_task)
                continue

            for etl_dir in etl_dirs:
                png_files = list(etl_dir.glob("*.png"))
                log_info(f"Found {len(png_files)} PNG files in {etl_dir}")

                for png_file in png_files:
                    txt_file = png_file.with_suffix(".txt")
                    if txt_file.exists():
                        pairs.append((png_file, txt_file))
            
            progress.advance(scan_task)

    if not pairs:
        log_error("No PNG/TXT pairs found!")
        return

    duration = datetime.now() - start_time
    log_info(f"\nFile pair search took: {duration}\nFound {len(pairs)} PNG/TXT pairs to process")

    if any(temp_dir.iterdir()):
        log_warning("Temporary directory is not empty. This could lead to duplicate data.")
        continue_input = input("Do you want to continue? (y/N): ").strip().lower()
        if continue_input != 'y':
            log_info("Operation cancelled by user.")
            return

    process_args = [(png, txt, temp_dir, i % args.workers) for i, (png, txt) in enumerate(pairs)]
    log_info(f"Processing with {args.workers} workers")

    results = []
    chunk_size = len(process_args) // (args.workers * 4)

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn()
            ) as progress:
                task = progress.add_task("🔄 Processing grids...", total=len(process_args))
                for result in executor.map(process_grid, process_args, chunksize=chunk_size):
                    results.append(result)
                    progress.advance(task)

    except KeyboardInterrupt:
        log_error("Processing interrupted by user")
        executor.shutdown(wait=False)
        return
    except Exception as e:
        log_error(f"Error during processing: {e}")
        executor.shutdown(wait=False)
        raise

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    table.add_row("Successfully Processed", str(success_count))
    table.add_row("Errors", str(error_count))
    table.add_row("Total Grids", str(len(pairs)))
    
    console.print("\n[bold]Processing Summary:[/]")
    console.print(table)

    if error_count > 0:
        console.print("\n[bold red]Failed grids:[/]")
        for result in results:
            if result["status"] == "error":
                console.print(f"  - {result['file']}: {result['error']}")

    # Save log file
    log_path = temp_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_pairs": len(pairs),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    log_info("✨ All done! Check the log file for detailed statistics.")

if __name__ == "__main__":
    main()
