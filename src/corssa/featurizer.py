import pandas as pd
import numpy as np

from skspatial.objects import Point, Vector, Plane
from scipy.spatial import distance_matrix

from corssa.parsers import parse_dssp

from scipy.spatial import distance_matrix

class Featurizer:
    def __init__(
            self,
            cg_df: pd.DataFrame,
            postfix='_rep'
        ):

        self.df = cg_df
        self.postfix = postfix

    def extract(
            self,
            calc_dist=True,
            calc_angles=True,
            calc_dihedrals=True,
            calc_neighbors=True,
            cat_features = ['aa', 'dssp'],
            remove_nan_aa = True
        ):
        """
        Process featurizer
        """
        if calc_dist: self.calc_dist_features()
        if calc_angles: self.calc_angle_features()
        if calc_dihedrals: self.calc_dihedral_angle_features()
        if calc_neighbors: self.calc_neighborhood_features()

        # Convert categorical features to category dtype
        for col in cat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')

        # Remove
        if remove_nan_aa:
            self.df = self.df[~self.df['aa'].isna()]

        return self.df

    def _dist_aa(self, start=0, end=0) -> float:
        """
        Calculate distance between atom and the next one.
        """

        cols = [f'x{self.postfix}', f'y{self.postfix}', f'z{self.postfix}']
        temp_cols = [f'_temp_{c}' for c in cols]

        temp_df = self.df.copy()

        temp_df[temp_cols] = temp_df[cols].shift(-start).diff(end-start)
        temp_df[temp_cols] = temp_df[temp_cols].pow(2)
        temp_df['dist'] = temp_df[temp_cols].sum(axis=1).pow(0.5)

        return temp_df['dist']

    def calc_dist_features(self):
        self.df[f'dist_1'] = self._dist_aa(end=1)
        self.df[f'dist_2'] = self._dist_aa(end=2)
        self.df[f'dist_-1'] = self._dist_aa(end=-1)
        self.df[f'dist_-2'] = self._dist_aa(end=-2)
        self.df[f'dist_-1_1'] = self._dist_aa(start=-1, end=1)
        self.df[f'dist_-1_2'] = self._dist_aa(start=-1, end=2)
        self.df[f'dist_-2_1'] = self._dist_aa(start=-2, end=1)
        self.df[f'dist_-2_2'] = self._dist_aa(start=-2, end=2)

    def _angle(self, p1, p2, p3):
        cols = [f'x{self.postfix}', f'y{self.postfix}', f'z{self.postfix}']
        temp_df = self.df[cols].copy()

        temp_df[['p1_x', 'p1_y', 'p1_z']] = self.df[cols].shift(-p1)
        temp_df[['p2_x', 'p2_y', 'p2_z']] = self.df[cols].shift(-p2)
        temp_df[['p3_x', 'p3_y', 'p3_z']] = self.df[cols].shift(-p3)

        temp_df.dropna(inplace=True)

        temp_df['point2'] = temp_df.apply(lambda row: Point([row['p2_x'], row['p2_y'], row['p2_z']]), axis=1)

        temp_df['v1'] = temp_df.apply(lambda row: Vector.from_points(
            row['point2'],
            Point([row['p1_x'], row['p1_y'], row['p1_z']])
        ), axis=1)

        temp_df['v2'] = temp_df.apply(lambda row: Vector.from_points(
            row['point2'],
            Point([row['p3_x'], row['p3_y'], row['p3_z']])
        ), axis=1)

        temp_df['plane'] = temp_df.apply(lambda row: Plane.from_vectors(
            row['point2'],
            row['v1'],
            row['v2']
        ), axis=1)
        temp_df['n_v1'] = temp_df.apply(lambda row: row['plane'].normal, axis=1)
        temp_df['angle'] = temp_df.apply(lambda row: np.degrees(row['v1'].angle_signed_3d(row['v2'], direction_positive=row['n_v1'])), axis=1)

        # Recover indexing
        for bottom_idx in range(0, -min(p1, p2, p3)):
            temp_df.loc[bottom_idx, 'angle'] = None

        for top_idx in range(len(self.df)-max(p1, p2, p3), len(self.df)):
            temp_df.loc[top_idx, 'angle'] = None

        return temp_df[['angle']].sort_index()

    def _dihedral(self, p1, p2, p3, p4):

        cols = [f'x{self.postfix}', f'y{self.postfix}', f'z{self.postfix}']
        temp_df = self.df[cols].copy()

        temp_df[['p1_x', 'p1_y', 'p1_z']] = self.df[cols].shift(-p1)
        temp_df[['p2_x', 'p2_y', 'p2_z']] = self.df[cols].shift(-p2)
        temp_df[['p3_x', 'p3_y', 'p3_z']] = self.df[cols].shift(-p3)
        temp_df[['p4_x', 'p4_y', 'p4_z']] = self.df[cols].shift(-p4)

        temp_df.dropna(inplace=True)

        temp_df['v1'] = temp_df.apply(lambda row: Vector.from_points(
            Point([row['p2_x'], row['p2_y'], row['p2_z']]),
            Point([row['p1_x'], row['p1_y'], row['p1_z']])
        ), axis=1)


        temp_df['v2'] = temp_df.apply(lambda row: Vector.from_points(
            Point([row['p2_x'], row['p2_y'], row['p2_z']]),
            Point([row['p3_x'], row['p3_y'], row['p3_z']])
        ), axis=1)

        temp_df['v3'] = temp_df.apply(lambda row: Vector.from_points(
            Point([row['p3_x'], row['p3_y'], row['p3_z']]),
            Point([row['p4_x'], row['p4_y'], row['p4_z']])
        ), axis=1)

        # # Normal vectors to the planes
        temp_df['n1'] = temp_df.apply(lambda row: row['v1'].cross(row['v2']), axis=1)
        temp_df['n2'] = temp_df.apply(lambda row: row['v2'].cross(row['v3']), axis=1)

        # # Normalize
        temp_df['n1_u'] = temp_df.apply(lambda row: row['n1'].unit(), axis=1)
        temp_df['n2_u'] = temp_df.apply(lambda row: row['n2'].unit(), axis=1)
        temp_df['v2_u'] = temp_df.apply(lambda row: row['v2'].unit(), axis=1)

        # Compute angle using atan2 for signed dihedral
        temp_df['plane_1'] = temp_df.apply(lambda row: row['n1_u'].dot(row['n2_u']), axis=1)
        temp_df['plane_2'] = temp_df.apply(lambda row: row['v2_u'].dot(row['n1_u'].cross(row['n2_u'])), axis=1)
        temp_df['dihedral_angle'] = temp_df.apply(lambda row: np.degrees(np.arctan2(row['plane_2'], row['plane_1'])), axis=1)

        # Recover indexing
        for bottom_idx in range(0, -min(p1, p2, p3, p4)):
            temp_df.loc[bottom_idx, 'dihedral_angle'] = None

        for top_idx in range(len(self.df)-max(p1, p2, p3, p4), len(self.df)):
            temp_df.loc[top_idx, 'dihedral_angle'] = None

        return temp_df[['dihedral_angle']].sort_index()

    def calc_angle_features(self):
        self.df[f'angle_-2-10'] = self._angle(-2, -1, 0)
        self.df[f'angle_-201'] = self._angle(-2, 0, 1)
        self.df[f'angle_-202'] = self._angle(-2, 0, 2)
        self.df[f'angle_-101'] = self._angle(-1, 0, 1)
        self.df[f'angle_-102'] = self._angle(-1, 0, 2)
        self.df[f'angle_012'] = self._angle(0, 1, 2)

    def calc_dihedral_angle_features(self):
        self.df[f'dihedral_-2-1'] = self._dihedral(-2, -1, 0, 1)
        self.df[f'dihedral_-1-2'] = self._dihedral(-1, 0, 1, 2)

    def _count_neighbors(self, distance_matrix, threshold):
        n = (distance_matrix < threshold).sum(axis=1) - 3  # Exclude self and immediate neighbors
        n[n < 0 ] = 0
        return n

    def calc_neighborhood_features(self):
        dist_arr = self.df[[f'x{self.postfix}', f'y{self.postfix}', f'z{self.postfix}']]
        dist_m = distance_matrix(dist_arr, dist_arr)

        self.df[f'neighbor_count_6A'] = self._count_neighbors(dist_m, 6.0)
        self.df[f'neighbor_count_7A'] = self._count_neighbors(dist_m, 7.0)
        self.df[f'neighbor_count_8A'] = self._count_neighbors(dist_m, 8.0)
        self.df[f'neighbor_count_9A'] = self._count_neighbors(dist_m, 9.0)
        self.df[f'neighbor_count_10A'] = self._count_neighbors(dist_m, 10.0)