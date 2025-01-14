from pathlib import Path
from tqdm import tqdm
import os
import random
import math
from PIL import Image
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import unicodedata
from rich.console import Console
from rich.prompt import Confirm, IntPrompt
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

console = Console()


def display_header():
    console.print(
        Panel.fit(
            "[bold cyan]Dataset Splitter Tool[/bold cyan]\n"
            "[dim]A tool for splitting and processing image datasets[/dim]",
            border_style="blue",
        )
    )


def get_user_percentage():
    while True:
        try:
            percent = IntPrompt.ask(
                "[yellow]Enter the percentage of dataset to use[/yellow]",
                default=100,
                show_default=True,
            )
            if 0 < percent <= 100:
                return percent / 100
            console.print("[red]Please enter a number between 1 and 100[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def is_cjk(char):
    ranges = [
        (0x4E00, 0x9FFF),
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0x3400, 0x4DBF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2F800, 0x2FA1F),
    ]
    code = ord(char)
    return any(start <= code <= end for start, end in ranges)


def get_processing_options():
    options = {}

    options["upscale"] = Confirm.ask(
        "[yellow]Do you want to upscale images?[/yellow]", default=False
    )

    if options["upscale"]:
        console.print("\n[cyan]Available target sizes:[/cyan]")
        console.print("1: 224x224")
        console.print("2: 384x384")

        while True:
            size = console.input("[yellow]Choose target size (1/2): [/yellow]")
            if size in ["1", "2"]:
                options["target_size"] = 224 if size == "1" else 384
                break
            console.print("[red]Invalid choice. Please select 1 or 2.[/red]")

        console.print("\n[cyan]Upscaling methods:[/cyan]")
        console.print("1: Lanczos (Recommended)")
        console.print("2: Bicubic")
        console.print("3: Bilinear")

        while True:
            method = console.input("[yellow]Choose upscaling method (1-3): [/yellow]")
            if method in ["1", "2", "3"]:
                options["method"] = {
                    "1": cv2.INTER_LANCZOS4,
                    "2": cv2.INTER_CUBIC,
                    "3": cv2.INTER_LINEAR,
                }[method]
                break
            console.print("[red]Invalid choice. Please select 1, 2, or 3.[/red]")

    options["normalize"] = Confirm.ask(
        "[yellow]Do you want to normalize images? (this will divide by 255)[/yellow]",
        default=False,
    )

    options["binarize"] = Confirm.ask(
        "[yellow]Do you want to binarize images using Otsu's method?[/yellow]",
        default=False,
    )

    options["invert"] = Confirm.ask(
        "[yellow]Do you want to invert the images?[/yellow]", default=False
    )

    return options


def confirm_choices(subset_percentage, source_dir, output_path, processing_options):
    console.print("\n[bold green]Please confirm your choices:[/bold green]")
    console.print(f"• Dataset Size: [cyan]{subset_percentage*100}%[/cyan]")
    console.print(f"• Source: [cyan]{source_dir}[/cyan]")
    console.print(f"• Output: [cyan]{output_path}[/cyan]")

    if processing_options["upscale"]:
        console.print(
            f"• Upscaling to: [cyan]{processing_options['target_size']}x{processing_options['target_size']}[/cyan]"
        )
    if processing_options["normalize"]:
        console.print("• Image normalization: [cyan]Enabled[/cyan]")
    if processing_options["cjk_only"]:
        console.print("• CJK characters only: [cyan]Enabled[/cyan]")
    if processing_options["binarize"]:
        console.print("• Binarization (Otsu): [cyan]Enabled[/cyan]")
    if processing_options["invert"]:
        console.print("• Image inversion: [cyan]Enabled[/cyan]")

    source_path = Path(source_dir)
    char_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if char_dirs:
        random_char = random.choice(char_dirs)
        cmd = f"python tools/grid_walk.py {random_char}/0.png {random_char.name} --solo --label {random_char.name}"
        console.print(f"\n[cyan]Demonstrating random character with:[/cyan] {cmd}")
        os.system(cmd)

    proceed = Confirm.ask(
        "\n[yellow]Proceed with these settings?[/yellow]", default=True
    )

    return proceed


def create_directory_structure(base_dir):
    dirs = {
        "train": {"images": base_dir / "train"},
        "val": {"images": base_dir / "val"},
        "test": {"images": base_dir / "test"},
    }

    for split in dirs.values():
        for dir_path in split.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def rename_files_in_dir(dir_path):
    files = sorted(list(dir_path.glob("*.png")))
    for idx, file in enumerate(files):
        new_name = f"{idx}.png"
        new_path = file.parent / new_name
        file.rename(new_path)

        label_file = Path(str(file).replace(".png", ".txt"))
        if label_file.exists():
            new_label_path = Path(str(new_path).replace(".png", ".txt"))
            label_file.rename(new_label_path)


def process_image(img_path, options):
    if not any(
        [
            options["upscale"],
            options["normalize"],
            options["binarize"],
            options["invert"],
        ]
    ):
        return None

    img = cv2.imread(str(img_path))

    if options["upscale"]:
        size = (options["target_size"], options["target_size"])
        img = cv2.resize(img, size, interpolation=options["method"])

    if options["normalize"]:
        img = img.astype("float32") / 255.0
        img = (img * 255).astype("uint8")

    if options["binarize"]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if options["invert"]:
        img = cv2.bitwise_not(img)

    return img


def copy_file(src, dst, options=None):
    if options is None or not (options["upscale"] or options["normalize"]):
        src_stat = os.stat(src)
        size = src_stat.st_size
        with open(src, "rb") as fsrc:
            with open(dst, "wb") as fdst:
                os.sendfile(fdst.fileno(), fsrc.fileno(), 0, size)
        os.utime(dst, (src_stat.st_atime, src_stat.st_mtime))
    else:
        img = process_image(src, options)
        if img is not None:
            cv2.imwrite(str(dst), img)
        else:
            print("Something went wrong with copying")


def normalize_char(char):
    if char == "ィ":
        return "ヰ"
    return unicodedata.normalize("NFKC", char)


def is_fullwidth_latin(char):

    # Full-width Latin
    ranges = [(0xFF21, 0xFF3A), (0xFF41, 0xFF5A), (0xFF10, 0xFF19)]
    code = ord(char)
    return any(start <= code <= end for start, end in ranges)


def process_character(args):
    char_dir, dirs, subset_percentage, processing_options = args

    char = char_dir.name if char_dir.name else ""

    if processing_options.get("cjk_only", False) and not is_cjk(char):
        return

    normalized_name = normalize_char(char)

    if not is_fullwidth_latin(char) and normalized_name == char:

        return process_character_normal(args)

    parent_dir = char_dir.parent
    matching_dirs = [
        d
        for d in parent_dir.iterdir()
        if d.is_dir() and normalize_char(d.name) == normalized_name
    ]

    all_images = []

    if not matching_dirs:
        print(f"No matching directories found for {char_dir.name}")
        return

    for dir in matching_dirs:
        all_images.extend(list(dir.glob("*.png")))

    if not all_images:
        print(f"No images found for {char_dir.name}")
        return

    total_subset = math.ceil(len(all_images) * subset_percentage)
    if len(all_images) < total_subset:
        print(
            f"Skipping {char_dir.name} - not enough samples ({len(all_images)} < {total_subset})"
        )
        return

    selected_images = random.sample(all_images, total_subset)
    if len(selected_images) < 3:
        # * If less than 3 images, resample 3, this used to just duplicate 1 image three times and thats not right
        # * realistically this will only ever matter if you use 1% of the dataset but eh
        selected_images = random.choices(all_images, k=3)

    if total_subset < 3:
        train_size = val_size = test_size = 1
    else:
        val_size = max(1, math.ceil(total_subset * 0.1))
        test_size = max(1, math.ceil(total_subset * 0.1))
        train_size = max(1, total_subset - val_size - test_size)

    splits = {
        "train": selected_images[:train_size],
        "val": selected_images[train_size : train_size + val_size],
        "test": selected_images[train_size + val_size :],
    }

    for split_name, images in splits.items():
        if not images:
            print(f"No images for {split_name} split in {char_dir.name}")

            continue

        dest_path = dirs[split_name]["images"] / normalized_name
        dest_path.mkdir(exist_ok=True)

        for i, img_path in enumerate(images):
            dst_path = dest_path / f"{i}.png"
            copy_file(str(img_path), str(dst_path), processing_options)


def process_character_normal(args):
    """this processes non fullwidth"""
    char_dir, dirs, subset_percentage, processing_options = args

    images = list(char_dir.glob("*.png"))

    total_subset = max(3, math.ceil(len(images) * subset_percentage))
    selected_images = random.sample(images, min(len(images), total_subset))
    if len(selected_images) < 3:
        # * If less than 3 images, resample 3, this used to just duplicate 1 image three times and thats not right
        # * realistically this will only ever matter if you use 1% of the dataset but eh
        selected_images = random.choices(images, k=3)

    if total_subset < 3:
        train_size = val_size = test_size = 1
    else:
        val_size = max(1, math.floor(total_subset * 0.1))
        test_size = max(1, math.floor(total_subset * 0.1))
        train_size = total_subset - val_size - test_size

    splits = {
        "train": selected_images[:train_size],
        "val": selected_images[train_size : train_size + val_size],
        "test": selected_images[train_size + val_size :],
    }

    for split_name, images in splits.items():
        if not images:
            print(f"No images for {split_name} split in {char_dir.name}")
            continue

        dest_path = dirs[split_name]["images"] / char_dir.name
        dest_path.mkdir(exist_ok=True)

        for i, img_path in enumerate(images):
            dst_path = dest_path / f"{i}.png"
            copy_file(str(img_path), str(dst_path), processing_options)


def split_dataset(source_dir, output_dir, subset_percentage, processing_options):
    source_dir = Path(source_dir)
    output_base = Path(output_dir) / f"dataset_{int(subset_percentage * 100)}percent"

    dirs = create_directory_structure(output_base)

    characters = [d for d in source_dir.iterdir() if d.is_dir()]

    print(f"Processing {len(characters)} characters using {cpu_count()} processes...")

    worker_args = [
        (char_dir, dirs, subset_percentage, processing_options)
        for char_dir in characters
    ]

    with Pool(processes=os.cpu_count()) as pool:
        list(
            tqdm(
                pool.imap(process_character, worker_args),
                total=len(characters),
                desc="Processing characters",
            )
        )


if __name__ == "__main__":
    FINAL_DIR = "etl_dataset"

    display_header()

    processing_options = get_processing_options()
    subset_percentage = get_user_percentage()
    output_dir = Path(FINAL_DIR).parent
    output_path = (
        output_dir
        / f"dataset_{int(subset_percentage*100)}percent_{'filtered' if processing_options['cjk_only'] else 'whole'}"
    )
    print(output_dir)

    if not confirm_choices(
        subset_percentage, FINAL_DIR, output_path, processing_options
    ):
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        exit()

    console.print("\n[bold yellow]Starting dataset split...[/bold yellow]")
    with console.status("[bold green]Processing dataset...[/bold green]"):
        split_dataset(FINAL_DIR, output_dir, subset_percentage, processing_options)

    console.print("\n[bold green]✨ Dataset split complete![/bold green]")
    console.print(f"Dataset saved to: [cyan]{output_path}[/cyan]")
