import argparse
import multiprocessing
import os
import pickle
import threading
from collections import OrderedDict

import joblib
import numpy as np
from matminer.featurizers.site import CoordinationNumber, CrystalNNFingerprint
from matminer.featurizers.site import OPSiteFingerprint, AGNIFingerprints
from matminer.utils.data import PymatgenData, MagpieData
from monty.io import zopen
from pymatgen import SymmOp
from pymatgen.io.cif import CifParser, CifBlock, CifFile
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from config import SCALER, MODELS_DIR
from matfeaturizers import GaussianSymmFuncModified, VoronoiFingerprintModified

PYMATGEN_FEATURES = [
    "X", "row", "group", "block", "atomic_mass", "atomic_radius",
    "mendeleev_no", "velocity_of_sound", "thermal_conductivity",
    "melting_point"
]

MAGPIE_FEATURES = [
    "Number", "MendeleevNumber", "AtomicWeight", "MeltingT",
    "Column", "Row", "CovalentRadius", "Electronegativity",
    "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
    "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled",
    "NUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom",
    "SpaceGroupNumber"
]


class CifWriterCharger:
    def __init__(self, struct, symprec=None, charges=None):
        """
        A wrapper around CifFile to write CIF files from pymatgen structures.

        Args:
            struct (Structure): structure to write
            symprec (float): If not none, finds the symmetry of the structure
                and writes the cif with symmetry information. Passes symprec
                to the SpacegroupAnalyzer
            write_magmoms (bool): If True, will write magCIF file. Incompatible
                with symprec
        """

        format_str = "{:.8f}"

        block = OrderedDict()
        loops = []
        spacegroup = ("P 1", 1)
        if symprec is not None:
            sf = SpacegroupAnalyzer(struct, symprec)
            spacegroup = (sf.get_space_group_symbol(),
                          sf.get_space_group_number())
            # Needs the refined struture when using symprec. This converts
            # primitive to conventional structures, the standard for CIF.
            struct = sf.get_refined_structure()

        latt = struct.lattice
        comp = struct.composition
        no_oxi_comp = comp.element_composition
        block["_symmetry_space_group_name_H-M"] = spacegroup[0]
        for cell_attr in ['a', 'b', 'c']:
            block["_cell_length_" + cell_attr] = format_str.format(
                getattr(latt, cell_attr))
        for cell_attr in ['alpha', 'beta', 'gamma']:
            block["_cell_angle_" + cell_attr] = format_str.format(
                getattr(latt, cell_attr))
        block["_symmetry_Int_Tables_number"] = spacegroup[1]
        block["_chemical_formula_structural"] = no_oxi_comp.reduced_formula
        block["_chemical_formula_sum"] = no_oxi_comp.formula
        block["_cell_volume"] = "%.8f" % latt.volume

        reduced_comp, fu = no_oxi_comp.get_reduced_composition_and_factor()
        block["_cell_formula_units_Z"] = str(int(fu))

        if symprec is None:
            block["_symmetry_equiv_pos_site_id"] = ["1"]
            block["_symmetry_equiv_pos_as_xyz"] = ["x, y, z"]
        else:
            sf = SpacegroupAnalyzer(struct, symprec)

            symmops = []
            for op in sf.get_symmetry_operations():
                v = op.translation_vector
                symmops.append(SymmOp.from_rotation_and_translation(
                    op.rotation_matrix, v))

            ops = [op.as_xyz_string() for op in symmops]
            block["_symmetry_equiv_pos_site_id"] = \
                ["%d" % i for i in range(1, len(ops) + 1)]
            block["_symmetry_equiv_pos_as_xyz"] = ops

        loops.append(["_symmetry_equiv_pos_site_id",
                      "_symmetry_equiv_pos_as_xyz"])

        try:
            symbol_to_oxinum = OrderedDict([
                (el.__str__(),
                 float(el.oxi_state))
                for el in sorted(comp.elements)])
            block["_atom_type_symbol"] = symbol_to_oxinum.keys()
            block["_atom_type_oxidation_number"] = symbol_to_oxinum.values()
            loops.append(["_atom_type_symbol", "_atom_type_oxidation_number"])
        except (TypeError, AttributeError):
            symbol_to_oxinum = OrderedDict([(el.symbol, 0) for el in
                                            sorted(comp.elements)])

        atom_site_type_symbol = []
        atom_site_symmetry_multiplicity = []
        atom_site_fract_x = []
        atom_site_fract_y = []
        atom_site_fract_z = []
        atom_site_label = []
        atom_site_occupancy = []
        atom_site_charge_label = []

        count = 1
        if symprec is None:
            for site in struct:
                for sp, occu in sorted(site.species_and_occu.items()):
                    atom_site_type_symbol.append(sp.__str__())
                    atom_site_symmetry_multiplicity.append("1")
                    atom_site_fract_x.append("{0:f}".format(site.a))
                    atom_site_fract_y.append("{0:f}".format(site.b))
                    atom_site_fract_z.append("{0:f}".format(site.c))
                    atom_site_label.append("{}{}".format(sp.symbol, count))
                    atom_site_occupancy.append(occu.__str__())

                    count += 1
        else:
            # The following just presents a deterministic ordering.
            unique_sites = [
                (sorted(sites, key=lambda s: tuple([abs(x) for x in
                                                    s.frac_coords]))[0],
                 len(sites))
                for sites in sf.get_symmetrized_structure().equivalent_sites
            ]
            for site, mult in sorted(
                    unique_sites,
                    key=lambda t: (t[0].species_and_occu.average_electroneg,
                                   -t[1], t[0].a, t[0].b, t[0].c)):
                for sp, occu in site.species_and_occu.items():
                    atom_site_type_symbol.append(sp.__str__())
                    atom_site_symmetry_multiplicity.append("%d" % mult)
                    atom_site_fract_x.append("{0:f}".format(site.a))
                    atom_site_fract_y.append("{0:f}".format(site.b))
                    atom_site_fract_z.append("{0:f}".format(site.c))
                    atom_site_label.append("{}{}".format(sp.symbol, count))
                    atom_site_occupancy.append(occu.__str__())
                    count += 1

        block["_atom_site_type_symbol"] = atom_site_type_symbol
        block["_atom_site_label"] = atom_site_label
        block["_atom_site_symmetry_multiplicity"] = \
            atom_site_symmetry_multiplicity
        block["_atom_site_fract_x"] = atom_site_fract_x
        block["_atom_site_fract_y"] = atom_site_fract_y
        block["_atom_site_fract_z"] = atom_site_fract_z
        block["_atom_site_occupancy"] = atom_site_occupancy
        block["_atom_site_charge"] = charges

        loops.append(["_atom_site_type_symbol",
                      "_atom_site_label",
                      "_atom_site_symmetry_multiplicity",
                      "_atom_site_fract_x",
                      "_atom_site_fract_y",
                      "_atom_site_fract_z",
                      "_atom_site_occupancy",
                      "_atom_site_charge",
                      ])

        d = OrderedDict()
        d[comp.reduced_formula] = CifBlock(block, loops, comp.reduced_formula)
        self._cf = CifFile(d)

    def __str__(self):
        """
        Returns the cif as a string.
        """
        return self._cf.__str__()

    def write_file(self, filename):
        """
        Write the cif file.
        """
        with zopen(filename, "wt") as f:
            f.write(self.__str__())


def get_fps(structure, cutoff=10.0, processes=8):
    all_descrs = []

    try:
        coordination_number_ = CoordinationNumber.from_preset('VoronoiNN')
        voronoi_fps_ = VoronoiFingerprintModified(cutoff=cutoff).featurize_structure(structure)
        crystal_nn_fingerprint_ = CrystalNNFingerprint.from_preset('cn')
        op_site_fingerprint_ = OPSiteFingerprint()
        agni_fingerprints_ = AGNIFingerprints()
        gaussian_symm_func_fps_ = GaussianSymmFuncModified().featurize_structure(structure)
        pymatgen_data_ = PymatgenData()
        magpie_data_ = MagpieData()

        data_list = [[structure, i, site, coordination_number_, voronoi_fps_,
                      crystal_nn_fingerprint_, op_site_fingerprint_,
                      agni_fingerprints_, gaussian_symm_func_fps_,
                      pymatgen_data_, magpie_data_] for i, site in enumerate(structure)]

        pool = multiprocessing.Pool(processes=processes)
        all_descrs = np.array(pool.map(get_all_site_descrs, data_list))

    except (AttributeError, IndexError) as error:
        pass

    return all_descrs


def get_all_site_descrs(data_list):
    structure = data_list[0]
    i = data_list[1]
    site = data_list[2]
    coordination_number_ = data_list[3]
    voronoi_fps_ = data_list[4]
    crystal_nn_fingerprint_ = data_list[5]
    op_site_fingerprint_ = data_list[6]
    agni_fingerprints_ = data_list[7]
    gaussian_symm_func_fps_ = data_list[8]
    pymatgen_data_ = data_list[9]
    magpie_data_ = data_list[10]

    coordination_number = coordination_number_.featurize(structure, i)
    voronoi_fingerprint = voronoi_fps_[i]
    crystal_nn_fingerprint = crystal_nn_fingerprint_.featurize(structure, i)
    op_site_fingerprint = op_site_fingerprint_.featurize(structure, i).tolist()
    agni_fingerprints = agni_fingerprints_.featurize(structure, i).tolist()
    gaussian_symm_func = gaussian_symm_func_fps_[i]
    pymatgen_data = [pymatgen_data_.get_elemental_property(site.specie, attr) for attr in PYMATGEN_FEATURES]
    magpie_data = [magpie_data_.get_elemental_property(site.specie, attr) for attr in MAGPIE_FEATURES]

    descrs = coordination_number + voronoi_fingerprint + crystal_nn_fingerprint \
             + op_site_fingerprint + agni_fingerprints + gaussian_symm_func + pymatgen_data + magpie_data

    return descrs


tls = threading.local()


def get_models(models_directory: str) -> dict:
    """Cache models in TLS for the case when server is run in threads mode"""
    models = getattr(tls, 'models', None)
    if models is not None:
        return models

    models = {}
    for model_filename in os.listdir(models_directory):
        filename = os.fsdecode(model_filename)
        if filename.endswith(".pickle.dat") and filename not in models:
            file = os.path.join(models_directory, filename)
            model = pickle.load(open(file, "rb"))
            models[filename] = model

    tls.models = models
    return models


def get_charges(structure_filename, output_filename):
    scaler = os.path.join(os.path.dirname(__file__), MODELS_DIR, SCALER)
    models_directory = os.path.join(os.path.dirname(__file__), MODELS_DIR)

    structures = CifParser(structure_filename).get_structures(primitive=False)
    if len(structures) == 0:
        raise AssertionError("No structures!")
    
    structure = structures[0]
    if not structure.is_ordered:
        raise AssertionError("Structure is non-stoichiometric - not supported yet!")

    scaler_ = joblib.load(scaler)
    scaled_features = scaler_.transform(np.delete(get_fps(structure), [139], axis=1))

    models = get_models(models_directory)

    predict_all = []
    for model in models.values():
        predict_all.append(model.predict(scaled_features))

    predict_all = np.mean(np.vstack(predict_all), axis=0)
    diff = sum(predict_all) / predict_all.shape[0]
    predict_all = np.around(predict_all - diff, decimals=4)

    cw = CifWriterCharger(structure, charges=predict_all)
    cw.write_file(output_filename)


if __name__ == '__main__':
    """ A basic command line interface for predicting MOF charges """

    parser = argparse.ArgumentParser(description="MOF charges prediction CLI")
    parser.add_argument("-i", "--input", type=str, help="input CIF file", required=True)
    parser.add_argument("-o", "--output", type=str, help="output CIF file")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input[:-4] + "-charges.cif"

    get_charges(args.input, args.output)
