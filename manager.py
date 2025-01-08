import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn
from rich import print as rprint
import time
import sys
import os
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import shutil


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.unpack import (
    fast_etl_file_scan,
    process_etl_file,
    load_jis_map,
    JISMappingMixin,
)
from tools.grid_walk import grid_walk
from tools.tiles_from_pairs import process_grid, ETL_IMAGE_SIZES
from tools.fast_merge import merge_worker_outputs
from tools.dataset_splitter import (
    split_dataset,
    get_processing_options,
    get_user_percentage,
    confirm_choices,
)

ETL_URLS = [
    <links here>
]

app = typer.Typer()
console = Console()


def validate_paths():
    """Validate required paths and files exist"""
    script_dir = Path(__file__).parent

    required_paths = {
        "JIS0201.TXT": script_dir / "mappings/JIS0201.TXT",
        "JIS0208.TXT": script_dir / "mappings/JIS0208.TXT",
        "euc_co59.dat": script_dir / "mappings/euc_co59.dat",
    }

    missing = []
    for name, path in required_paths.items():
        if not path.exists():
            missing.append(name)

    if missing:
        console.print(f"[red]Missing required files: {', '.join(missing)}[/red]")
        console.print(
            "[yellow]Please ensure all mapping files are in the mappings directory[/yellow]"
        )
        raise typer.Exit(1)


def download_file(url, output_dir: Path, progress_bar=None):
    try:
        filename = url.split("/")[-1]
        download_path = output_dir / "download_dir" / filename
        extract_path = output_dir / "helper/raw_dataset"

        download_path.parent.mkdir(parents=True, exist_ok=True)
        extract_path.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Download failed with status code: {response.status_code}")

        with open(download_path, "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)

        if filename.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

        return {"filename": filename, "success": True}
    except Exception as e:
        return {"filename": url.split("/")[-1], "success": False, "error": str(e)}


@app.command()
def download(
    output_dir: Path = typer.Argument(..., help="Directory to store downloaded files"),
    workers: int = typer.Option(4, help="Number of concurrent downloads"),
):

    console.print(
        Panel.fit(
            f"[cyan]ETL Download[/cyan]\nOutput: {output_dir}\nWorkers: {workers}",
            title="Stage 0: Downloading",
            border_style="blue",
        )
    )

    if not Confirm.ask("Start downloading ETL files?"):
        raise typer.Exit()

    failed_downloads = []

    with console.status(
        "[blue]Downloading ETL files...[/blue]", spinner="dots"
    ) as status:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for url in ETL_URLS:
                futures.append(executor.submit(download_file, url, output_dir))

            for future in futures:
                try:
                    result = future.result()
                    if result["success"]:
                        console.print(
                            f"[green]✓[/green] Successfully processed {result['filename']}"
                        )
                    else:
                        failed_downloads.append(result)
                        console.print(
                            f"[red]✗[/red] Failed to process {result['filename']}: {result['error']}"
                        )
                except Exception as e:
                    console.print(f"[red]✗[/red] Unexpected error: {str(e)}")
                    failed_downloads.append({"filename": "Unknown", "error": str(e)})

    if failed_downloads:
        console.print(
            Panel.fit(
                f"[red]Warning: {len(failed_downloads)} downloads failed![/red]\n"
                + "\n".join(
                    [f"- {f['filename']}: {f['error']}" for f in failed_downloads]
                ),
                border_style="red",
            )
        )
        if not Confirm.ask("Continue despite failed downloads?"):
            raise typer.Exit(1)

    return output_dir / "helper/raw_dataset"


@app.command()
def unpack(
    input_dir: Path = typer.Argument(..., help="Directory containing ETL files"),
    workers: int = typer.Option(os.cpu_count(), help="Number of worker processes"),
):
    validate_paths()

    console.print(
        Panel.fit(
            f"[cyan]ETL Unpacking[/cyan]\nInput: {input_dir}\nWorkers: {os.cpu_count()}",
            title="Stage 1: Unpacking",
            border_style="blue",
        )
    )

    if not input_dir.exists():
        console.print(f"[red]Input directory {input_dir} does not exist![/red]")
        raise typer.Exit(1)

    output_dir = input_dir
    output_dir.mkdir(exist_ok=True)

    existing_files = list(input_dir.glob("*/*.png")) + list(input_dir.glob("*/*.txt"))
    if existing_files:
        if Confirm.ask("Found existing unpacked files. Skip unpacking?"):
            return output_dir
    elif not Confirm.ask("Start unpacking?"):
        raise typer.Exit()

    mapping_201 = load_jis_map("mappings/JIS0201.TXT", format="201")
    mapping_208 = load_jis_map("mappings/JIS0208.TXT", format="208")
    JISMappingMixin.set_mapping(mapping_201, mapping_208)

    etl_files = fast_etl_file_scan(str(input_dir))
    if not etl_files:
        console.print("[red]No ETL files found in input directory[/red]")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:
        task = progress.add_task("Unpacking ETL files...", total=len(etl_files))

        with multiprocessing.Pool(workers) as pool:
            process_args = [
                (str(f), "mappings/JIS0201.TXT", "mappings/JIS0208.TXT", False)
                for f in etl_files
            ]

            for result in pool.imap_unordered(process_etl_file, process_args):
                progress.advance(task)

    console.print("[green]Unpacking complete![/green]")
    return output_dir


@app.command()
def process_grids(
    grid_dir: Path = typer.Argument(
        ..., help="Directory containing unpacked ETL grids"
    ),
    workers: int = typer.Option(os.cpu_count(), help="Number of worker processes"),
):

    console.print(
        Panel.fit(
            f"[cyan]Grid Processing[/cyan]\nInput: {grid_dir}\nWorkers: {workers}",
            title="Stage 2: Grid Processing",
            border_style="blue",
        )
    )
    console.print("[yellow]Syncing filesystem...[/yellow]")
    os.sync()

    temp_dir = grid_dir.parent / "temp_workers"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not grid_dir.exists():
        console.print(f"[red]Grid directory {grid_dir} does not exist![/red]")
        raise typer.Exit(1)

    if list(temp_dir.glob("worker_*")):
        if Confirm.ask("Found existing grid processing outputs. Skip processing?"):
            return temp_dir

    if not Confirm.ask("Start processing grids?"):
        raise typer.Exit()

    pairs = []
    with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:
        task = progress.add_task("Scanning for PNG/TXT pairs...", total=None)

        for etl_type in ETL_IMAGE_SIZES.keys():
            etl_dirs = list(grid_dir.glob(f"*{etl_type}*"))
            for etl_dir in etl_dirs:
                png_files = list(etl_dir.glob("*.png"))
                for png_file in png_files:
                    txt_file = png_file.with_suffix(".txt")
                    if txt_file.exists():
                        pairs.append((png_file, txt_file))

    if not pairs:
        console.print("[red]No PNG/TXT pairs found![/red]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(pairs)} PNG/TXT pairs to process[/green]")

    process_args = [
        (png, txt, temp_dir, i % workers) for i, (png, txt) in enumerate(pairs)
    ]
    results = []

    with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:
        task = progress.add_task("Processing grids...", total=len(process_args))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            chunk_size = max(1, len(process_args) // (workers * 4))

            try:
                for result in executor.map(
                    process_grid, process_args, chunksize=chunk_size
                ):
                    results.append(result)
                    progress.advance(task)
            except Exception as e:
                console.print(f"[red]Error during processing: {str(e)}[/red]")
                raise typer.Exit(1)

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    console.print(
        Panel.fit(
            f"[green]Successfully processed: {success_count}[/green]\n"
            f"[red]Errors: {error_count}[/red]\n"
            f"Total grids: {len(pairs)}",
            title="Processing Summary",
            border_style="blue",
        )
    )

    if error_count > 0:
        console.print("\n[red]Failed grids:[/red]")
        for result in results:
            if result["status"] == "error":
                console.print(f"  - {result['file']}: {result['error']}")

        if not Confirm.ask("Continue despite errors?"):
            raise typer.Exit(1)

    console.print("[green]Grid processing complete![/green]")
    return temp_dir


@app.command()
def merge(
    temp_dir: Path = typer.Argument(
        ..., help="Directory containing processed grid chunks"
    ),
    workers: int = typer.Option(8, help="Number of worker processes"),
):

    console.print(
        Panel.fit(
            f"[cyan]Merging Chunks[/cyan]\nInput: {temp_dir}\nWorkers: {workers}",
            title="Stage 3: Merging",
            border_style="blue",
        )
    )
    console.print("[yellow]Syncing filesystem...[/yellow]")
    os.sync()

    if not temp_dir.exists():
        console.print(f"[red]Temp directory {temp_dir} does not exist![/red]")
        raise typer.Exit(1)

    if not Confirm.ask("Start merging?"):
        raise typer.Exit()

    final_dir = temp_dir.parent.parent / "final"
    final_dir.mkdir(exist_ok=True)

    merge_worker_outputs(
        dry_run=False,
        output_dir=str(temp_dir),
        final_dir=str(final_dir),
        num_workers=workers,
    )

    console.print("[green]Merging complete![/green]")
    return final_dir


@app.command()
def split(
    final_dir: Path = typer.Argument(..., help="Directory containing merged dataset"),
    cjk_only: bool = typer.Option(
        False, help="Keep only CJK characters in the final dataset"
    ),
):
    """Split final dataset into train/val/test sets"""
    console.print(
        Panel.fit(
            f"[cyan]Dataset Splitting[/cyan]\nInput: {final_dir}",
            title="Stage 4: Splitting",
            border_style="blue",
        )
    )

    console.print("[yellow]Syncing filesystem...[/yellow]")
    os.sync()

    if not final_dir.exists():
        console.print(f"[red]Final directory {final_dir} does not exist![/red]")
        raise typer.Exit(1)

    processing_options = get_processing_options()
    subset_percentage = get_user_percentage()
    output_dir = (
        final_dir.parent
        / f"dataset_{int(subset_percentage*100)}percent_{'filtered' if cjk_only else 'whole'}"
    )

    if not confirm_choices(
        subset_percentage, final_dir, output_dir, processing_options
    ):
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit()

    if output_dir.exists():
        if Confirm.ask(
            f"\n[yellow]Output directory {output_dir} already exists. Delete it?[/yellow]"
        ):
            shutil.rmtree(output_dir)
        else:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            raise typer.Exit()

    if not Confirm.ask("Start splitting dataset?"):
        raise typer.Exit()

    split_dataset(final_dir, output_dir, subset_percentage, processing_options)

    console.print("[green]Dataset splitting complete![/green]")


@app.command()
def pipeline(
    base_dir: Path = typer.Option(
        Path.cwd() / "etl_dataset", help="Base directory for all pipeline outputs"
    ),
    workers: int = typer.Option(8, help="Number of worker processes"),
    cjk_only: bool = typer.Option(
        False, help="Keep only CJK characters in the final dataset"
    ),
):
    start_time = time.time()

    dirs = {
        "download": base_dir / "download_dir",
        "helper": {
            "temp": base_dir / "helper/temp_workers",
            "raw": base_dir / "helper/raw_dataset",
        },
        "final": base_dir / "final",
    }

    for path in [
        dirs["download"],
        dirs["helper"]["temp"],
        dirs["helper"]["raw"],
        dirs["final"],
    ]:
        path.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "[cyan]ETL Dataset Processing Pipeline[/cyan]\n"
            f"Output Directory: {base_dir}\n"
            "This will run all processing stages in sequence",
            title="Pipeline Started",
            border_style="blue",
        )
    )

    try:
        raw_dir = dirs["helper"]["raw"]
        dl = dirs["download"]
        if Confirm.ask("Would you like to download ETL files?"):
            raw_dir = download(base_dir, workers=4)
        elif not list(dl.glob("ETL*")):
            console.print(
                "[red]No ETL files found in raw directory. Cannot proceed without downloads.[/red]"
            )
            raise typer.Exit(1)

        grid_dir = unpack(raw_dir, workers)
        temp_dir = process_grids(grid_dir, workers)
        if not list(temp_dir.glob("worker_*")):
            raise Exception("No processed files found after grid processing")

        final_dir = dirs["final"]
        if list(final_dir.glob("*")):
            if not Confirm.ask("Files found in final dir, skip merging?"):
                final_dir = merge(temp_dir, workers)
        elif not list(final_dir.glob("*")):
            console.print(
                "[red]No merged files found in final directory. Cannot proceed without merging.[/red]"
            )
            if Confirm.ask("Would you like to merge now?"):
                final_dir = merge(temp_dir, workers)
            else:
                raise typer.Exit(1)

        split(final_dir, cjk_only=cjk_only)

        elapsed = time.time() - start_time
        console.print(f"\n[green]Pipeline complete! Total time: {elapsed:.2f}s[/green]")

    except Exception as e:
        console.print(
            Panel.fit(f"[red]Pipeline failed![/red]\n{str(e)}", border_style="red")
        )
        raise typer.Exit(1)
        
@app.command()
def debug_grid(
    image_path: Path = typer.Argument(..., help="Path to grid image file"),
    txt_path: Path = typer.Argument(..., help="Path to corresponding label text file"),
):

    if not image_path.exists():
        console.print(f"[red]Image file {image_path} does not exist![/red]")
        raise typer.Exit(1)

    if not txt_path.exists():
        console.print(f"[red]Label file {txt_path} does not exist![/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[cyan]Grid Debug Viewer[/cyan]\nImage: {image_path}\nLabels: {txt_path}",
            title="Grid Debugger",
            border_style="blue",
        )
    )

    try:
        grid_walk(image_path, txt_path)
    except Exception as e:
        console.print(f"[red]Error during grid debugging: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
