import pandas as pd

from Bio.PDB.Polypeptide import PPBuilder
from Bio.Data.IUPACData import atom_weights

from corssa.parsers import parse_cif, parse_dssp
from tqdm.contrib.concurrent import thread_map

class CoarseGrainModel:
    def __init__(self, postfix='_rep'):
        self.postfix = postfix

    def scheme(self, structure, postfix, **kwargs):
        """
        Coarse graining scheme. Should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    def process(self, cif_filepath: str, dssp_filepath: str = None, export_cols: list = [], **kwargs) -> pd.DataFrame:
        """
        Process coarse graining according to scheme. Return Pandas DataFrame
        """

        structure = parse_cif(cif_filepath)

        residues = [
            res.get_resname()
            for res in structure.get_residues()
        ]

        aa_df = pd.DataFrame(
            data=residues,
            columns=['aa']
        )

        structure_df = self.scheme(structure, self.postfix, **kwargs)

        df = pd.concat([aa_df, structure_df], axis=1)

        outcols = [
            'aa',
            f'x{self.postfix}',
            f'y{self.postfix}',
            f'z{self.postfix}',
            *export_cols
        ]

        # Load DSSP data if provided
        if dssp_filepath is not None:
            df_dssp = parse_dssp(dssp_filepath)
            df = pd.concat([df, df_dssp[['dssp']]], axis=1)
            outcols.append('dssp')

        return df[outcols]

    def process_batch(self, cif_filepaths: list, dssp_filepaths: list = None, export_cols: list = [], **kwargs) -> pd.DataFrame:
        """
        Process batch of CIF files. Return combined Pandas DataFrame.
        """

        if dssp_filepaths is None:
            dssp_filepaths = [None] * len(cif_filepaths)

        def _process_single(payload):
            cif, dssp = payload
            return self.process(cif, dssp_filepath=dssp, export_cols=export_cols, **kwargs)

        dfs = thread_map(_process_single, zip(cif_filepaths, dssp_filepaths))

        return pd.concat(dfs, ignore_index=True)


class CAlphaRep(CoarseGrainModel):
    def __init__(self, postfix='_ca'):
        super().__init__(postfix=postfix)

    def scheme(self, structure, postfix, **kwargs):
        """
        Coarse graining scheme. Should be implemented in subclasses.
        """

        ppbuilder = PPBuilder()
        p = ppbuilder.build_peptides(structure)[0]

        coordinates = [
            a.get_coord() for a in p.get_ca_list()
        ]

        df = pd.DataFrame(
            data=coordinates,
            columns=[f"x{postfix}", f"y{postfix}", f"z{postfix}"]
        )

        return df

class CBetaRep(CoarseGrainModel):
    def __init__(self, postfix='_cb'):
        super().__init__(postfix=postfix)

    def _get_cb_coord(residue):
        try:
            return residue['CB'].get_coord()
        except KeyError:
            return residue['CA'].get_coord()

    def scheme(self, structure, postfix, **kwargs):
        """
        Coarse graining scheme. Should be implemented in subclasses.
        """
        ppbuilder = PPBuilder()
        p = ppbuilder.build_peptides(structure)[0]

        coordinates = map(self._get_cb_coord, p)

        df = pd.DataFrame(
            data=coordinates,
            columns=[f"x{postfix}", f"y{postfix}", f"z{postfix}"]
        )

        return df

class CCOMRep(CoarseGrainModel):
    def __init__(self, postfix='_com'):
        super().__init__(postfix=postfix)

    def _residue_center_of_mass(self, residue):
        total_mass = 0.0
        weighted_sum = [0.0, 0.0, 0.0]

        for atom in residue:
            name = atom.element  # Biopython stores element symbol
            mass = atom_weights.get(name, 0.0)
            x, y, z = atom.coord

            total_mass += mass
            weighted_sum[0] += mass * x
            weighted_sum[1] += mass * y
            weighted_sum[2] += mass * z

        return [c / total_mass for c in weighted_sum]

    def scheme(self, structure, postfix, **kwargs):
        """
        Coarse graining scheme. Should be implemented in subclasses.
        """

        ppbuilder = PPBuilder()
        p = ppbuilder.build_peptides(structure)[0]

        coordinates = map(self._residue_center_of_mass, p)

        df = pd.DataFrame(
            data=coordinates,
            columns=[f"x{postfix}", f"y{postfix}", f"z{postfix}"]
        )

        return df