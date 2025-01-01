import codecs
import bitstring  # type: ignore
from PIL import Image  # type: ignore
from pathlib import Path
import argparse
import csv
import re
import jaconv  # type: ignore
import multiprocessing
from tqdm import tqdm  # type: ignore
import codecs
import logging
import sys
import unicodedata

def jis_to_hiragana(text): #! mainly obsolete, you might be able to just hardcode this into ETL7 
    if not text:
        return None
    #* JIS x 0201 mapped files contain half-width katakana (from old terminal intefaces). 
    #* normalising through NFKC will convert half-width katakana to full-width
    #* then we subtract the offset (from UTF docs) to convert to hiragana
    #* we dont care about predicting half-width katakana so this should be fine
    is_halfwidth = (0xFF61 <= ord(text) <= 0xFF9F)
    
    if is_halfwidth:
        x = unicodedata.normalize('NFKC', text).replace('ヴ', 'う゛')
        if 0x30A1 <= ord(x) <= 0x30F3:
            return chr(ord(x) - 0x60)  # Convert to hiragana
    
    return text
   
   

def load_jis_map(*filenames):
    jis_to_unicode = {}
    
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        jis_code = int(parts[0].replace("0x", ""), 16)
                        unicode_value = int(parts[1].replace("0x", ""), 16)
                        jis_to_unicode[jis_code] = unicode_value
    return jis_to_unicode


class CO59_to_utf8:
    def __init__(self, euc_co59_file="euc_co59.dat"):
        with codecs.open(euc_co59_file, "r", "euc-jp") as f:
            co59t = f.read()
        co59l = co59t.split()
        self.conv = {}
        for c in co59l:
            ch = c.split(":")
            co = ch[1].split(",")
            co59c = (int(co[0]), int(co[1]))
            self.conv[co59c] = ch[0]

    def __call__(self, co59):
        return self.conv[co59]


def T56(c):
    t56s = "0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);'|/STUVWXYZ ,%=\"!"
    return t56s[c]


class JISMappingMixin:  # ?
    _jis_mapping_201 = None
    _jis_mapping_208 = None

    @classmethod
    def set_mapping(cls, mapping_201, mapping_208=None):
        cls._jis_mapping_201 = mapping_201
        cls._jis_mapping_208 = mapping_208

    @classmethod
    def get_mapping(cls, mapping_type='201'):
        if mapping_type == '201':
            if cls._jis_mapping_201 is None:
                raise RuntimeError("JIS0201 mapping not initialized. Call set_mapping first.")
            return cls._jis_mapping_201
        elif mapping_type == '208':
            if cls._jis_mapping_208 is None:
                raise RuntimeError("JIS0208 mapping not initialized. Call set_mapping first.")
            return cls._jis_mapping_208
        raise ValueError("Invalid mapping type. Use '201' or '208'")


class ETLn_Record:
    def read(self, bs, pos=None):
        if pos:
            f.bytepos = pos * self.octets_per_record

        r = bs.readlist(self.bitstring)

        record = dict(zip(self.fields, r))

        self.record = {
            k: (self.converter[k](v) if k in self.converter else v)
            for k, v in record.items()
        }

        return self.record

    def get_image(self):
        return self.record["Image Data"]


class ETL167_Record(ETLn_Record, JISMappingMixin):  # * For ETL1 and ETL6
    def __init__(self):
        self.octets_per_record = 2052
        self.fields = [
            "Data Number",
            "Character Code",
            "Serial Sheet Number",
            "JIS Code",
            "EBCDIC Code",
            "Evaluation of Individual Character Image",
            "Evaluation of Character Group",
            "Male-Female Code",
            "Age of Writer",
            "Serial Data Number",
            "Industry Classification Code",
            "Occupation Classification Code",
            "Sheet Gatherring Date",
            "Scanning Date",
            "Sample Position Y on Sheet",
            "Sample Position X on Sheet",
            "Minimum Scanned Level",
            "Maximum Scanned Level",
            "Image Data",
        ]
        self.bitstring = "uint:16,bytes:2,uint:16,hex:8,hex:8,4*uint:8,uint:32,4*uint:16,4*uint:8,pad:32,bytes:2016,pad:32"
        self.converter = {
            "Character Code": lambda x: x.decode("ascii"),
            "Image Data": lambda x: Image.eval(
                Image.frombytes("F", (64, 63), x, "bit", 4).convert("L"),
                lambda x: x * 16,
            ),
        }

    def get_char(self):
        """get & convert JIS code to Unicode character - normalizes to full-width katakana only""" #? explained in readme
        jis_code = self.record["JIS Code"]
        if isinstance(jis_code, str):
            jis_code = int(jis_code.replace("0x", ""), 16)

        if jis_code == 0x0:
            return chr(0x0000)

        unicode_value = None
        unicode_value = self.get_mapping('201').get(jis_code)
        
        if unicode_value is None:
            logging.error(f"No Unicode mapping found for JIS code: {hex(jis_code)}")
            return None
            
        char = chr(unicode_value)
        #* Only normalize to full-width if half-width
        if 0xFF61 <= ord(char) <= 0xFF9F:
            return unicodedata.normalize('NFKC', char).replace('ヴ', 'ウ゛')
        
        return char

class ETL7_Record(ETL167_Record):  #? check readme
    def get_char(self):
        """get & convert JIS code to Unicode character - normalizes and converts to hiragana"""
        jis_code = self.record["JIS Code"]
        if isinstance(jis_code, str):
            jis_code = int(jis_code.replace("0x", ""), 16)

        if jis_code == 0x0:
            return chr(0x0000)

        unicode_value = None
        unicode_value = self.get_mapping('201').get(jis_code)
        
        if unicode_value is None:
            logging.error(f"No Unicode mapping found for JIS code: {hex(jis_code)}")
            return None
            
        char = chr(unicode_value)
        return jis_to_hiragana(char)


class ETL2_Record(ETLn_Record):  # * by my testing works, co59 should just work, if it bugs out check euc_co59.dat maybe?
    def __init__(self):
        self.octets_per_record = 2745
        self.fields = [
            "Serial Data Number",
            "Mark of Style",
            "Contents",
            "Style",
            "CO-59 Code",
            "Image Data",
        ]
        self.bitstring = (
            "uint:36,uint:6,pad:30,bits:36,bits:36,pad:24,bits:12,pad:180,bytes:2700"
        )
        self.converter = {
            "Mark of Style": lambda x: T56(x),
            "Contents": lambda x: "".join([T56(b.uint) for b in x.cut(6)]),
            "CO-59 Code": lambda x: tuple([b.uint for b in x.cut(6)]),
            "Style": lambda x: "".join([T56(b.uint) for b in x.cut(6)]),
            "Image Data": lambda x: Image.eval(
                Image.frombytes("F", (60, 60), x, "bit", 6).convert("L"),
                lambda x: x * 4,
            ),
        }
        self.co59_to_utf8 = CO59_to_utf8("euc_co59.dat")

    def get_char(self):
        return self.co59_to_utf8(self.record["CO-59 Code"])


class ETL345_Record(ETLn_Record, JISMappingMixin):  # * works!
    def __init__(self):
        self.octets_per_record = 2952
        self.fields = [
            "Serial Data Number", "Serial Sheet Number", "JIS Code", "EBCDIC Code", "4 Character Code",
            "Evaluation of Individual Character Image", "Evaluation of Character Group",
            "Sample Position Y on Sheet", "Sample Position X on Sheet",
            "Male-Female Code", "Age of Writer", "Industry Classification Code", "Occupation Classification Code",
            "Sheet Gatherring Date", "Scanning Date", "Number of X-Axis Sampling Points",
            "Number of Y-Axis Sampling Points", "Number of Levels of Pixel",
            "Magnification of Scanning Lenz", "Serial Data Number (old)", "Image Data"        
        ]
        self.bitstring = 'uint:36,uint:36,hex:8,pad:28,hex:8,pad:28,bits:24,pad:12,15*uint:36,pad:1008,bytes:2736'
        self.converter = {
            '4 Character Code': lambda x: ''.join([ T56(b.uint) for b in x.cut(6) ]),
            'Image Data': lambda x: Image.eval(Image.frombytes('F', (72,76), x, 'bit', 4).convert('L'),
            lambda x: x * 16)
        }

    def get_char(self):
        jis_code = self.record["JIS Code"]
        if isinstance(jis_code, str):
            jis_code = int(jis_code.replace("0x", ""), 16)

        if jis_code == 0x0:
            return chr(0x0000)

        unicode_value = None
        unicode_value = self.get_mapping('201').get(jis_code)

        if unicode_value is None:
            logging.error(f"No Unicode mapping found for JIS code: {hex(jis_code)}")
            return None

        char = chr(unicode_value)
        
        #? Handle special character conversions based on T56
        if self.record['4 Character Code'][0] == 'H':
            char = jaconv.kata2hira(jaconv.han2zen(char))
            char = char.replace('ぃ', 'ゐ').replace('ぇ', 'ゑ')
        elif self.record['4 Character Code'][0] == 'K':
            char = jaconv.han2zen(char)
            char = char.replace('ィ', 'ヰ').replace('ェ', 'ヱ')

        return jis_to_hiragana(char)


class ETL8G_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 8199
        self.fields = [
            "Serial Sheet Number",
            "JIS Kanji Code",
            "JIS Typical Reading",
            "Serial Data Number",
            "Evaluation of Individual Character Image",
            "Evaluation of Character Group",
            "Male-Female Code",
            "Age of Writer",
            "Industry Classification Code",
            "Occupation Classification Code",
            "Sheet Gatherring Date",
            "Scanning Date",
            "Sample Position X on Sheet",
            "Sample Position Y on Sheet",
            "Image Data",
        ]
        self.bitstring = "uint:16,hex:16,bytes:8,uint:32,4*uint:8,4*uint:16,2*uint:8,pad:240,bytes:8128,pad:88"
        self.converter = {
            "JIS Typical Reading": lambda x: x.decode("ascii"),
            "Image Data": lambda x: Image.eval(
                Image.frombytes("F", (128, 127), x, "bit", 4).convert("L"),
                lambda x: x * 16,
            ),
        }

    def get_char(self):
        char = bytes.fromhex(
            "1b2442" + self.record["JIS Kanji Code"] + "1b2842"
        ).decode("iso2022_jp")
        return char


class ETL8B_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 512
        self.fields = [
            "Serial Sheet Number",
            "JIS Kanji Code",
            "JIS Typical Reading",
            "Image Data",
        ]
        self.bitstring = "uint:16,hex:16,bytes:4,bytes:504"
        self.converter = {
            "JIS Typical Reading": lambda x: x.decode("ascii"),
            "Image Data": lambda x: Image.frombytes("1", (64, 63), x, "raw"),
        }

    def get_char(self):
        char = bytes.fromhex(
            "1b2442" + self.record["JIS Kanji Code"] + "1b2842"
        ).decode("iso2022_jp")
        return char


class ETL9G_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 8199
        self.fields = [
            "Serial Sheet Number",
            "JIS Kanji Code",
            "JIS Typical Reading",
            "Serial Data Number",
            "Evaluation of Individual Character Image",
            "Evaluation of Character Group",
            "Male-Female Code",
            "Age of Writer",
            "Industry Classification Code",
            "Occupation Classification Code",
            "Sheet Gatherring Date",
            "Scanning Date",
            "Sample Position X on Sheet",
            "Sample Position Y on Sheet",
            "Image Data",
        ]
        self.bitstring = "uint:16,hex:16,bytes:8,uint:32,4*uint:8,4*uint:16,2*uint:8,pad:272,bytes:8128,pad:56"
        self.converter = {
            "JIS Typical Reading": lambda x: x.decode("ascii"),
            "Image Data": lambda x: Image.eval(
                Image.frombytes("F", (128, 127), x, "bit", 4).convert("L"),
                lambda x: x * 16,
            ),
        }

    def get_char(self):
        char = bytes.fromhex(
            "1b2442" + self.record["JIS Kanji Code"] + "1b2842"
        ).decode("iso2022_jp")
        return char


class ETL9B_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 576
        self.fields = [
            "Serial Sheet Number",
            "JIS Kanji Code",
            "JIS Typical Reading",
            "Image Data",
        ]
        self.bitstring = "uint:16,hex:16,bytes:4,bytes:504,pad:512"
        self.converter = {
            "JIS Typical Reading": lambda x: x.decode("shift_jis"),
            "Image Data": lambda x: Image.frombytes("1", (64, 63), x, "raw"),
        }

    def get_char(self):
        try:
            char = bytes.fromhex(
                "1b2442" + self.record["JIS Kanji Code"] + "1b2842"
            ).decode("iso2022_jp")
            return char
        except UnicodeDecodeError:
            logging.error(f"\nFailed to decode character: {char}\n")
            return "__null__"


def unpack(filename, etln_record):

    try:
        base = Path(filename).name
        folder = Path(filename).parent
        temp_files = []  # Track created files for cleanup on failure

        f = bitstring.ConstBitStream(filename=filename)

        if re.match(r"ETL[89]B_Record", etln_record.__class__.__name__):
            f.bytepos = etln_record.octets_per_record

        chars = []
        images = []
        records = []

        rows, cols = 40, 50
        rows_by_cols = rows * cols

        c = 0

        while f.pos < f.length:
            record = etln_record.read(f)
            #       print(f"Record: {record}")
            try:
                char = etln_record.get_char()
             #   print(f"gotten char: {char}")
                logging.debug(f"Position {f.pos}: Got character {char}")
            except Exception as e:
                logging.error(f"\nPosition {f.pos}: Failed to decode - {e}\n")
                char = "__null__"
            img = etln_record.get_image()

            chars.append(char)
            images.append(img)
            records.append(record)

            if len(chars) % rows_by_cols == 0 or f.pos >= f.length:
                txt = "\n".join(
                    ["".join(chars[j * cols : (j + 1) * cols]) for j in range(rows)]
                )
                txtfn = folder / "{}_{:02d}.txt".format(base, c)
                temp_files.append(txtfn)

                with open(txtfn, "w", encoding="utf-8") as txtf:
                    txtf.write(txt)

                w, h = images[0].width, images[0].height

                tiled = Image.new(images[0].mode, (w * cols, h * rows))

                for ij in range(len(images)):
                    i, j = ij % cols, ij // cols
                    tiled.paste(images[ij], (w * i, h * j))

                tiledfn = folder / "{}_{:02d}.png".format(base, c)
                temp_files.append(tiledfn)
                tiled.save(tiledfn)

                chars = []
                images = []
                c += 1

        csvfn = folder / "{}.csv".format(base)
        temp_files.append(csvfn)

        with open(csvfn, "w") as rf:
            writer = csv.writer(rf)
            writer.writerow(etln_record.fields[:-1])
            for ir in records:
                writer.writerow(list(ir.values())[:-1])

        return True, None

    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")
        logging.error(f"Record: {etln_record.read(f)}")
        for f in temp_files:
            try:
                if f.exists():
                    f.unlink()
            except:
                pass
        return False, str(e)


def process_etl_file(args):
    if isinstance(args, tuple):
        file_path, mapping_path_201, mapping_path_208 = args
    else:
        file_path = args
        mapping_path_201 = "/Users/chai/mojinet/utils/JIS0201.TXT"
        mapping_path_208 = "/Users/chai/mojinet/utils/JIS0208.TXT"

    base = Path(file_path).name
    #  print(f"Processing {base}...")

    etln_record = None
    needs_mapping = False

    if re.match(r"ETL[16]", base):
        etln_record = ETL167_Record()
        needs_mapping = True
    elif re.match(r"ETL7", base):
        etln_record = ETL7_Record()
        needs_mapping = True
    elif re.match(r"ETL2", base):
        etln_record = ETL2_Record()
    elif re.match(r"ETL[345]", base):
        etln_record = ETL345_Record()
        needs_mapping = True
    elif re.match(r"ETL8G", base):
        etln_record = ETL8G_Record()
    elif re.match(r"ETL8B", base):
        etln_record = ETL8B_Record()
    elif re.match(r"ETL9G", base):
        etln_record = ETL9G_Record()
    elif re.match(r"ETL9B", base):
        etln_record = ETL9B_Record()
    else:
        return (base, False, "Unknown ETL format")

    if needs_mapping:
        mapping_201 = load_jis_map(mapping_path_201)
        mapping_208 = load_jis_map(mapping_path_208)
        JISMappingMixin.set_mapping(mapping_201, mapping_208)

    success, error = unpack(str(file_path), etln_record)
    return (base, success, error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose ETL files")
    parser.add_argument("input", help="input directory containing ETL files")
    parser.add_argument("--workers", type=int, default=1, 
                      help="Number of worker processes (default: single process)")
    parser.add_argument("--single", action="store_true", default=False, 
                      help="Process a single file")
    parser.add_argument("--jis201", default="/Users/chai/mojinet/utils/JIS0201.TXT",
                      help="Path to JIS X 0201 mapping file")
    parser.add_argument("--jis208", default="/Users/chai/mojinet/utils/JIS0208.TXT",
                      help="Path to JIS X 0208 mapping file")
    args = parser.parse_args()

    #* Load both mappings
    mapping_201 = load_jis_map(args.jis201)
    mapping_208 = load_jis_map(args.jis208)
    JISMappingMixin.set_mapping(mapping_201, mapping_208)

    input_path = Path(args.input)
    print(f"Processing ETL files in {input_path}")
    logging.info(f"Processing ETL files in {input_path}")

    if args.single:
        etl_files = [input_path]
    else:
        etl_files = [f for f in input_path.glob("ETL*_*") if f.is_file() and f.suffix == ""]
    
    if not etl_files:
        print(f"No ETL files found in {input_path}")
        exit(1)

    print(f"Found {len(etl_files)} ETL files. Processing with {args.workers} workers...")

    with multiprocessing.Pool(args.workers) as pool:
        process_args = [(str(f), args.jis201, args.jis208) for f in etl_files]
        results = list(
            tqdm(
                pool.imap(process_etl_file, process_args),
                total=len(etl_files),
                desc="Processing ETL files 📝",
                bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} files [eta: {remaining}]",
            )
        )

    success_count = sum(1 for _, success, _ in results if success)
    print(
        f"\nProcessing complete: {success_count}/{len(etl_files)} files processed successfully"
    )

    failures = [(name, error) for name, success, error in results if not success]
    if failures:
        print("\nFailed files:")
        for name, error in failures:
            print(f"- {name}: {error}")
