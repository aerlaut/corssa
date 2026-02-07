from pathlib import Path
import re
import pandas as pd
from Bio.SeqUtils import seq3
from Bio.PDB import MMCIFParser

def parse_cif(filepath: str, structure_id: str = None):
    """
    Parser for the CIF file format. Returns the structure as a Bio.PDB.Structure object.

    Args:
        filepath (str): Path to the CIF file.
        structure_id (str, optional): Identifier for the structure. If None, it will be derived from the filename.
    """

    if structure_id is None:
        structure_id = Path(filepath).stem

    cif_parser = MMCIFParser(QUIET=True)
    structure = cif_parser.get_structure(structure_id, filepath)

    return structure

def parse_dssp(filepath: str):
    """
    Parser for the DSSP file format. Retrieves amino acid type, secondary structure and C-alpha coordinates.
    The data is returned as a pandas DataFrame with the columns "aa", "dssp", "x_ca", "y_ca", "z_ca".

    Args:
        filepath (str): Path to the DSSP file
    """

    def parse_line(line: str):
        aa = seq3(line[13]).upper()
        dssp = line[16].strip() or '.'
        x, y, z = map(float, re.sub(r"\s+", ",", line[115:].strip()).split(","))

        return {
            "aa": aa,
            "dssp": dssp,
            "x_ca": x,
            "y_ca": y,
            "z_ca": z
        }

    dssp = Path(filepath).read_text()
    lines = dssp.splitlines()[28:]

    return pd.DataFrame(map(parse_line, lines))