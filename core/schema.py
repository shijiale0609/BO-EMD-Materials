from __future__ import annotations

from collections import defaultdict

import pandas as pd
from loguru import logger
from pandas._typing import FilePath
from pydantic import BaseModel

from core.utils import json_dump, json_load


class Mixture(BaseModel):
    """ a mixture is a material consists of one or more molecular entities """

    components: list[str | None]
    """ a list of canonical SMILES """

    weights: list[float | None]
    """ roughly map to the concentrations of components"""

    additional_parameters: dict[str, float] = dict()
    """ things like temperature or reaction time """

    figure_of_merit: float
    """ optimization target """


class MixtureDataset(BaseModel):
    """ a mixture dataset is just a list of mixtures """

    mixture_list: list[Mixture]

    role_list: list[str]

    provenance: dict = {}

    def dump_as_json(self, filename: FilePath):
        json_dump(filename, self.model_dump(), indent=2)

    @classmethod
    def load_from_json(cls, filename: FilePath) -> MixtureDataset:
        dictionary = json_load(filename)
        return cls(**dictionary)

    @property
    def unique_components(self) -> list[str]:
        components = []
        for mix in self.mixture_list:
            components += mix.components
        return sorted(set(components))

    @property
    def component_domains(self) -> dict[str, list[str]]:
        """ possible smis for given roles"""
        domains = defaultdict(list)
        for mix in self.mixture_list:
            for role, comp in zip(self.role_list, mix.components):
                domains[role].append(comp)
        for role in domains:
            domains[role] = sorted(set(domains[role]), key=lambda x: (x is None, x))
        return domains

    @property
    def component_combination_size(self):
        s = 1
        for role in self.component_domains:
            s *= len(self.component_domains[role])
        return s

    @classmethod
    def from_csv(
            cls, csv: FilePath,
            smi_columns: list[int], weight_columns: list[int] | None, add_columns: list[int], fom: str | int,
            role_list: list[str],
    ) -> MixtureDataset:
        assert len(smi_columns), "no component column specified!"
        logger.info(f"we assume a list of {len(smi_columns)}-component mixtures")
        df = pd.read_csv(csv)

        if weight_columns is None:
            logger.info("no weight column is specified, use eq weights by default")
        else:
            assert len(smi_columns) == len(weight_columns)
            weight_columns = [df.columns[i] for i in weight_columns]

        smi_columns = [df.columns[i] for i in smi_columns]
        logger.info(f"loading smi columns: {smi_columns}")
        add_columns = [df.columns[i] for i in add_columns]

        if isinstance(fom, int):
            fom = df.columns[fom]

        mixtures = []
        for idx, row in df.iterrows():
            components = []
            for k in smi_columns:
                v = row[k]
                if pd.isna(v):
                    components.append(None)
                else:
                    components.append(v)
            if weight_columns is None:
                weights = [1 / len(components), ] * len(components)
                # n_valid_components = len([c for c in components if c is not None])
                # weights = [1 / n_valid_components, ] * n_valid_components
                # # TODO this is probably wrong if one of them is additive and only present in small amounts
            else:
                weights = [row[k] for k in weight_columns]
            additional_params = {k: row[k] for k in add_columns}
            fom_value = row[fom]
            mixture = Mixture(
                components=components, weights=weights, additional_parameters=additional_params,
                figure_of_merit=fom_value
            )
            mixtures.append(mixture)
        prov = dict(smi_columns=smi_columns, weight_columns=weight_columns, add_columns=add_columns, fom=fom,
                    role_list=role_list)
        return cls(mixture_list=mixtures, provenance=prov, role_list=role_list)
