from core.schema import MixtureDataset

"""
reaction datasets from
https://github.com/qai222/aspire_opt?tab=readme-ov-file#benchmark-reaction-datasets
"""


def dump_Buchwald_Hartwig():
    md = MixtureDataset.from_csv(
        "bdf1.csv",
        smi_columns=[6, 9, 12, 15],
        weight_columns=None, add_columns=[],
        fom=17,
        role_list=['base', 'ligand', 'halide', 'additive']
    )
    md.dump_as_json("reaction_bh.json")


def dump_Dehalogenation():
    md = MixtureDataset.from_csv(
        "bdf3.csv",
        smi_columns=[1, 2],
        weight_columns=None, add_columns=[],
        fom=3,
        role_list=['pc1', 'pc2']
    )
    md.dump_as_json("reaction_dh.json")


if __name__ == '__main__':
    dump_Buchwald_Hartwig()
    dump_Dehalogenation()
