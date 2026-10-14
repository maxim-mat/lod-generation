import logging

logger = logging.getLogger(__name__)


def _iter_surfaces(geom):
    """Yield (rings, semantic_type_or_None) for each surface of a geometry.

    ``rings`` is the surface as CityJSON stores it -- exterior first, interior
    rings after (spec 2.0.1). Only Solid / MultiSurface / CompositeSurface
    contribute; anything else warns rather than yielding nothing quietly.
    """
    boundaries = geom.get("boundaries", [])
    gtype = geom.get("type")
    sem = geom.get("semantics") or {}
    surfaces = sem.get("surfaces") or []
    values = sem.get("values") or []
    if gtype == "Solid":
        faces = [f for shell in boundaries for f in shell]
        vals = [v for shell_vals in values for v in shell_vals] if values else []
    elif gtype in ("MultiSurface", "CompositeSurface"):
        faces, vals = boundaries, values
    else:
        # Silently yielding nothing here once hid 265k lod-0 footprints; a
        # geometry the parser cannot read must say so.
        logger.warning("unsupported geometry type %r (lod %s): %d boundaries ignored",
                       gtype, geom.get("lod"), len(boundaries))
        return
    for i, face in enumerate(faces):
        if not face or len(face[0]) < 3:
            continue
        stype = None
        if i < len(vals) and vals[i] is not None and vals[i] < len(surfaces):
            s = surfaces[vals[i]]
            stype = s.get("type") if s else None
        yield face, stype
