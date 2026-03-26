# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

"""
Advanced Fenestration Controller
Radiance Frads Library module.

COPY FROM FRADS https://github.com/LBNL-ETA/frads/blob/main/frads/window.py#L590
"""

# pylint: skip-file

from frads.window import Gap, GlazingSystem, create_pwc_gaps, LayerInput, get_layer_data, Gas
from frads.window import _parse_input_source, NM_PER_MM
from frads.window import SHADING, VENETIAN, FABRIC
from frads.window import _apply_opening_properties, _process_blind_definition_to_bsdf
from frads.window import generate_melanopic_bsdf, Layer
import pywincalc as pwc


def create_glazing_system(
    name: str,
    layer_inputs: list[LayerInput],
    gaps: None | list[Gap] = None,
    nproc: int = 1,
    nsamp: int = 2000,
    mbsdf: bool = False,
    layer_data_only=False, # ADDED by Christoph
    bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL) # ADDED by Christoph
) -> GlazingSystem:
    """Create a glazing system from a list of layers and gaps.

    Args:
        name: Name of the glazing system.
        layer_inputs: List of layer inputs containing material specifications.
        gaps: List of gaps between layers (auto-generated if None).
        nproc: Number of processes for parallel computation.
        nsamp: Number of samples for Monte Carlo integration.
        mbsdf: Whether to generate melanopic BSDF data.

    Returns:
        GlazingSystem object containing optical and thermal properties.

    Raises:
        ValueError: Invalid layer type or input format.

    Examples:
        >>> from frads import LayerInput
        >>> layers = [
        ...     LayerInput("glass.json"),
        ...     LayerInput("venetian.xml")
        ... ]
        >>> gs = create_glazing_system("double_glazed", layers)
    """
    if gaps is None:
        gaps = [Gap([Gas("air", 1)], 0.0127) for _ in range(len(layer_inputs) - 1)]
    product_data_list: list[pwc.ProductData] = []
    layer_data: list[Layer] = []
    thickness = 0.0
    for idx, layer_inp in enumerate(layer_inputs):
        product_data = _parse_input_source(layer_inp.input_source)
        if product_data is None:
            raise ValueError("Invalid layer type")
        layer = get_layer_data(product_data)
        if product_data.product_type == SHADING:
            if product_data.product_subtype == VENETIAN:
                actual_product_data = _process_blind_definition_to_bsdf(
                    layer_inp, product_data, layer, nproc=nproc, nsamp=nsamp
                )
                layer_data.append(layer)
                product_data_list.append(actual_product_data)
                thickness += layer.thickness_m
            else:
                layer.product_type = FABRIC
                with open(layer_inp.input_source, "r") as f:
                    layer.shading_xml = f.read()
                layer_data.append(layer)
                product_data_list.append(product_data)
                thickness += layer.thickness_m
            _apply_opening_properties(
                layer, layer_inp.openings, gaps, idx, len(layer_inputs)
            )
        else:
            layer.spectral_data = {
                int(round(d.wavelength * NM_PER_MM)): (
                    d.direct_component.transmittance_front,
                    d.direct_component.reflectance_front,
                    d.direct_component.reflectance_back,
                )
                for d in product_data.measurements
            }
            layer.coated_side = product_data.coated_side
            layer_data.append(layer)
            product_data_list.append(product_data)
            thickness += layer.thickness_m
            
    if layer_data_only:
        return layer_data, product_data_list, thickness

    for gap in gaps:
        thickness += gap.thickness_m

    glzsys = pwc.GlazingSystem(
        solid_layers=product_data_list,
        gap_layers=create_pwc_gaps(gaps),
        width_meters=1,
        height_meters=1,
        environment=pwc.nfrc_shgc_environments(),
        bsdf_hemisphere=bsdf_hemisphere,
    )
    
    melanopic_back_transmittace = []
    melanopic_back_reflectance = []
    if mbsdf:
        melanopic_back_transmittace, melanopic_back_reflectance = (
            generate_melanopic_bsdf(layer_data, gaps, nproc=nproc, nsamp=nsamp)
        )

    # ALREADY flipping input file; ignore here
    #for index, data in enumerate(layer_data):
    #    if data.flipped:
    #        glzsys.flip_layer(index, True)

    solres = glzsys.optical_method_results("SOLAR")
    solsys = solres.system_results
    visres = glzsys.optical_method_results("PHOTOPIC")
    vissys = visres.system_results
    
    gs = GlazingSystem(
        name=name,
        thickness=thickness,
        layers=layer_data,
        gaps=gaps,
        solar_front_absorptance=[
            alpha.front.absorptance.angular_total for alpha in solres.layer_results
        ],
        solar_back_absorptance=[
            alpha.back.absorptance.angular_total for alpha in solres.layer_results
        ],
        visible_back_reflectance=vissys.back.reflectance.matrix,
        visible_front_reflectance=vissys.front.reflectance.matrix,
        visible_back_transmittance=vissys.back.transmittance.matrix,
        visible_front_transmittance=vissys.front.transmittance.matrix,
        solar_back_reflectance=solsys.back.reflectance.matrix,
        solar_front_reflectance=solsys.front.reflectance.matrix,
        solar_back_transmittance=solsys.back.transmittance.matrix,
        solar_front_transmittance=solsys.front.transmittance.matrix,
        melanopic_back_transmittance=melanopic_back_transmittace,
        melanopic_back_reflectance=melanopic_back_reflectance,
    )

    return gs, glzsys, solres, visres