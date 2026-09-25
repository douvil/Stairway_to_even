import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# 1. PARAMÈTRES
# ============================================================

DATA_DIR = "./regions/Annual/1901-2024"

FILE_TMP = f"{DATA_DIR}/tmp_cru_Annual_gll.nc"
FILE_PRE = f"{DATA_DIR}/pre_cru_Annual_gll.nc"
FILE_PET = f"{DATA_DIR}/pet_cru_Annual_gll.nc"

VAR_TMP = "tmp"
VAR_PRE = "pre"
VAR_PET = "pet"

OBS_START = 1901
OBS_END   = 2024
END_YEAR = 2035

REF_START = 1901
REF_END   = 1930


# ------------------------------------------------------------
# Fenêtres temporelles
# ------------------------------------------------------------

STEP_STARTS = np.arange(1901, 2007, 15)

STEP_WINDOWS = [
    (int(start), int(start + 29))
    for start in STEP_STARTS
]

N_STEPS = len(STEP_WINDOWS)


# ============================================================
# 2. PARAMÈTRES GÉOMÉTRIQUES
# ============================================================

THETA_DEG = 30.		# Angle en degré de l'axe profondeur
THETA = np.deg2rad(THETA_DEG)
BASE_STEP_WIDTH = 10.	# Largeur de la première marche
WIDTH_PCTRED = 4.0	# Réduction de largeur à chaque marche
STEP_HEIGHT = 1.00	# Hauteur verticale d'une marche
DEPTH_PER_YEAR = 0.075	# Échelle de profondeur

# Origine graphique
ORIGIN_X = 0.0
ORIGIN_Y = 0.0

# Contours
EDGE_COLOR = "black"
EDGE_LW = 0.45


# ============================================================
# 3. COLORMAPS
# ============================================================

CMAP_T = mpl.colormaps["RdBu_r"].copy()
CMAP_P = mpl.colormaps["BrBG"].copy()
CMAP_PET = mpl.colormaps["YlOrRd"].copy()

CMAP_T.set_bad("white")
CMAP_P.set_bad("white")
CMAP_PET.set_bad("white")


# ============================================================
# 4. DÉTECTION DES COORDONNÉES
# ============================================================

def find_coordinate(
    da,
    candidates,
    standard_names=None,
    axis=None
):

    candidates_lower = {
        str(name).lower()
        for name in candidates
    }

    standard_names = standard_names or []

    # Recherche par nom
    for name in da.coords:

        if str(name).lower() in candidates_lower:
            return name

    for name in da.dims:

        if str(name).lower() in candidates_lower:
            return name

    # Attributs CF
    for name in da.coords:

        coord = da.coords[name]

        standard_name = str(
            coord.attrs.get(
                "standard_name",
                ""
            )
        ).lower()

        coord_axis = str(
            coord.attrs.get(
                "axis",
                ""
            )
        ).upper()

        if standard_name in standard_names:
            return name

        if (
            axis is not None
            and coord_axis == axis
        ):
            return name

    raise ValueError(
        "\nCoordonnée introuvable.\n"
        f"Candidats : {candidates}\n"
        f"Dimensions : {list(da.dims)}\n"
        f"Coordonnées : {list(da.coords)}"
    )


def get_lat_name(da):

    return find_coordinate(
        da,
        [
            "lat",
            "latitude",
            "nav_lat",
            "y"
        ],
        ["latitude"],
        "Y"
    )


def get_lon_name(da):

    return find_coordinate(
        da,
        [
            "lon",
            "longitude",
            "nav_lon",
            "x"
        ],
        ["longitude"],
        "X"
    )


def get_time_name(da):

    return find_coordinate(
        da,
        [
            "time",
            "year"
        ],
        ["time"],
        "T"
    )


# ============================================================
# 5. OUTILS TEMPORELS
# ============================================================

def annual_years(da):

    time_name = get_time_name(da)

    time_coord = da[time_name]

    try:

        return (
            time_coord
            .dt.year
            .values
            .astype(int)
        )

    except Exception:

        return np.asarray(
            time_coord.values
        ).astype(int)


def subset_years(
    da,
    start_year,
    end_year
):

    time_name = get_time_name(da)

    years = annual_years(da)

    mask = (
        (years >= start_year)
        &
        (years <= end_year)
    )

    return da.isel(
        {
            time_name: np.where(mask)[0]
        }
    )


# ============================================================
# 6. MOYENNE GLOBALE
# ============================================================

def global_weighted_mean(da):
    """
    Si la variable possède uniquement la dimension time,
    elle est considérée comme déjà agrégée.

    Sinon : moyenne spatiale pondérée par cos(latitude).
    """

    time_name = get_time_name(da)

    spatial_dims = [
        dim
        for dim in da.dims
        if dim != time_name
    ]

    # --------------------------------------------------------
    # Données déjà agrégées
    # --------------------------------------------------------

    if len(spatial_dims) == 0:

        print(
            f"Variable déjà agrégée : {da.dims}"
        )

        return da


    # --------------------------------------------------------
    # Données spatialisées
    # --------------------------------------------------------

    lat_name = get_lat_name(da)

    latitude = da[lat_name]

    weights = np.cos(
        np.deg2rad(latitude)
    )

    weights.name = "latitude_weights"

    print(
        "Moyenne spatiale : "
        f"lat={lat_name}, "
        f"dims={spatial_dims}"
    )

    return da.weighted(weights).mean(
        dim=spatial_dims,
        skipna=True
    )


# ============================================================
# 7. LECTURE DES DONNÉES
# ============================================================

ds_t = xr.open_dataset(FILE_TMP)
ds_p = xr.open_dataset(FILE_PRE)
ds_pet = xr.open_dataset(FILE_PET)

tmp = ds_t[VAR_TMP]
pre = ds_p[VAR_PRE]
pet = ds_pet[VAR_PET]


print("\n===== STRUCTURE DES VARIABLES =====")

for name, var in [
    ("TMP", tmp),
    ("PRE", pre),
    ("PET", pet)
]:

    print(
        f"{name}: "
        f"dims={var.dims}, "
        f"shape={var.shape}"
    )


# ============================================================
# 8. PÉRIODE OBSERVÉE
# ============================================================

tmp = subset_years(
    tmp,
    OBS_START,
    OBS_END
)

pre = subset_years(
    pre,
    OBS_START,
    OBS_END
)

pet = subset_years(
    pet,
    OBS_START,
    OBS_END
)


# ============================================================
# 9. SÉRIES CLIMATIQUES
# ============================================================

T_global = global_weighted_mean(tmp)

P_global = global_weighted_mean(pre)

PET_global = global_weighted_mean(pet)


# ============================================================
# 10. SÉRIES ANNUELLES 1901–2035
# ============================================================

years_observed = annual_years(
    T_global
)

T_observed = np.asarray(
    T_global.values,
    dtype=float
).squeeze()

P_observed = np.asarray(
    P_global.values,
    dtype=float
).squeeze()

PET_observed = np.asarray(
    PET_global.values,
    dtype=float
).squeeze()


all_years = np.arange(
    OBS_START,
    END_YEAR + 1
)


T = np.full(
    len(all_years),
    np.nan
)

P = np.full(
    len(all_years),
    np.nan
)

PET = np.full(
    len(all_years),
    np.nan
)


for (
    year,
    temperature,
    precipitation,
    aridity
) in zip(
    years_observed,
    T_observed,
    P_observed,
    PET_observed
):

    index = int(
        year - OBS_START
    )

    if 0 <= index < len(all_years):

        T[index] = temperature
        P[index] = precipitation
        PET[index] = aridity


# ============================================================
# 11. ANOMALIES 1901–1930
# ============================================================

reference_mask = (
    (all_years >= REF_START)
    &
    (all_years <= REF_END)
)


T_reference = np.nanmean(
    T[reference_mask]
)

P_reference = np.nanmean(
    P[reference_mask]
)

PET_reference = np.nanmean(
    PET[reference_mask]
)


dT = T - T_reference
dP = P - P_reference
dPET = PET - PET_reference


# ============================================================
# 12. NORMALISATION
# ============================================================

def symmetric_limit(
    values,
    percentile=98
):

    values = np.asarray(values)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 1.0

    limit = np.nanpercentile(
        np.abs(values),
        percentile
    )

    if (
        not np.isfinite(limit)
        or limit == 0
    ):
        return 1.0

    return float(limit)


nobs = OBS_END - OBS_START + 1


limit_T = symmetric_limit(
    dT[:nobs]
)

limit_P = symmetric_limit(
    dP[:nobs]
)

limit_PET = symmetric_limit(
    dPET[:nobs]
)


norm_T = TwoSlopeNorm(
    vmin=-limit_T,
    vcenter=0.0,
    vmax=limit_T
)

norm_P = TwoSlopeNorm(
    vmin=-limit_P,
    vcenter=0.0,
    vmax=limit_P
)

norm_PET = TwoSlopeNorm(
    vmin=-limit_PET,
    vcenter=0.0,
    vmax=limit_PET
)


# ============================================================
# 13. LARGEUR DES MARCHES
# ============================================================

def step_width(step_index):
    """
    Largeur diminuant progressivement.

    step_index = 0 -> première marche
    step_index = 7 -> dernière marche
    """

    factor = (
        1.0
        -
        WIDTH_PCTRED / 100.0
    ) ** step_index

    return (
        BASE_STEP_WIDTH
        * factor
    )


def step_right(step_index):
    """
    Le bord droit est volontairement conservé au même
    emplacement géométrique.

    Cela maintient un flanc PET continu.
    """

    return BASE_STEP_WIDTH


def step_left(step_index):
    """
    La réduction de largeur se fait vers l'intérieur,
    donc le bord gauche se déplace progressivement
    vers la droite.
    """

    return (
        step_right(step_index)
        -
        step_width(step_index)
    )


print("\n===== LARGEUR DES MARCHES =====")

for i in range(N_STEPS):

    print(
        f"Marche {i + 1}: "
        f"{step_width(i):.3f}"
    )


# ============================================================
# 14. GÉOMÉTRIE DE PROJECTION
# ============================================================

# Axe horizontal
U = np.array([
    1.0,
    0.0
])


# Axe oblique PET / profondeur
V = DEPTH_PER_YEAR * np.array([
    np.cos(THETA),
    np.sin(THETA)
])


# Verticale
H = np.array([
    0.0,
    1.0
])


ORIGIN = np.array([
    ORIGIN_X,
    ORIGIN_Y
])


def project(
    x,
    year,
    height
):

    year_offset = (
        year - OBS_START
    )

    return (
        ORIGIN
        + x * U
        + year_offset * V
        + height * H
    )


# ============================================================
# 15. UTILITAIRES GRAPHIQUES
# ============================================================

def add_polygon(
    ax,
    points,
    **kwargs
):

    polygon = Polygon(
        np.asarray(points),
        closed=True,
        **kwargs
    )

    ax.add_patch(
        polygon
    )

    return polygon


def value_for_year(
    values,
    year
):

    index = int(
        year - OBS_START
    )

    if 0 <= index < len(values):
        return values[index]

    return np.nan


def color_for_value(
    cmap,
    norm,
    value
):

    if not np.isfinite(value):

        return (
            1.0,
            1.0,
            1.0,
            1.0
        )

    c = cmap(
        norm(value)
    )

    # Alpha forcé à 1
    return (
        c[0],
        c[1],
        c[2],
        1.0
    )


# ============================================================
# 16. PROFIL DU FLANC PET
# ============================================================

def stair_height(year):

    n = np.sum(
        STEP_STARTS <= year
    )

    n = np.clip(
        n,
        1,
        N_STEPS
    )

    return (
        n * STEP_HEIGHT
    )


# ============================================================
# 17. FIGURE A4 PAYSAGE
# ============================================================

fig = plt.figure(
    figsize=(11.69, 8.27),
    facecolor="white"
)


ax = fig.add_axes([
    0.045,
    0.19,
    0.91,
    0.74
])


ax.set_aspect(
    "equal"
)

ax.axis(
    "off"
)

ax.set_facecolor(
    "white"
)


# ============================================================
# 18. MARCHES : P + SAT
# ============================================================

#
# Arrière -> avant
#
for step_index in reversed(
    range(N_STEPS)
):

    start_year, end_year = (
        STEP_WINDOWS[
            step_index
        ]
    )

    height_bottom = (
        step_index
        * STEP_HEIGHT
    )

    height_top = (
        (step_index + 1)
        * STEP_HEIGHT
    )


    # --------------------------------------------------------
    # Géométrie horizontale de cette marche
    # --------------------------------------------------------

    x_left = step_left(
        step_index
    )

    x_right = step_right(
        step_index
    )

    width = step_width(
        step_index
    )


    # --------------------------------------------------------
    # Profondeur complète de 30 ans
    # --------------------------------------------------------

    depth_start = start_year
    depth_end = end_year + 1


    # ========================================================
    # 18a. DESSUS : PRÉCIPITATIONS
    # ========================================================

    for annual_index, year in enumerate(
        range(
            start_year,
            end_year + 1
        )
    ):

        fraction0 = (
            annual_index / 30.0
        )

        fraction1 = (
            (annual_index + 1) / 30.0
        )


        x0 = (
            x_left
            + fraction0 * width
        )

        x1 = (
            x_left
            + fraction1 * width
        )

        # Bord gauche modifié : il suit la liaison vers la
        # façade SAT de la marche suivante. Au-delà du début
        # de la marche suivante, il reste à son abscisse.
        if step_index < N_STEPS - 1:
            next_start_year = STEP_WINDOWS[
                step_index + 1
            ][0]
            next_x_left = step_left(
                step_index + 1
            )

            def left_boundary_x(depth):
                progress = (
                    depth - depth_start
                ) / (
                    next_start_year - depth_start
                )
                progress = min(
                    max(progress, 0.0),
                    1.0
                )
                return (
                    x_left
                    + progress * (
                        next_x_left - x_left
                    )
                )

            x0_left = left_boundary_x(depth_start)
            x1_left = left_boundary_x(depth_end)
        else:
            x0_left = x_left
            x1_left = x_left

        # Si la nouvelle limite dépasse entièrement la bande,
        # celle-ci est ignorée plutôt que de créer un polygone
        # inversé ou débordant.
        if x1 <= max(x0_left, x1_left):
            continue


        points = [

            project(
                max(x0, x0_left),
                depth_start,
                height_top
            ),

            project(
                x1,
                depth_start,
                height_top
            ),

            project(
                x1,
                depth_end,
                height_top
            ),

            project(
                max(x0, x1_left),
                depth_end,
                height_top
            )

        ]


        value = value_for_year(
            dP,
            year
        )


        add_polygon(
            ax,
            points,
            facecolor=color_for_value(
                CMAP_P,
                norm_P,
                value
            ),
            edgecolor="none",
            linewidth=0,
            alpha=1.0,
            antialiased=False,
            zorder=10 + step_index
        )


    # --------------------------------------------------------
    # Contour du dessus, sans l'ancien bord gauche
    # --------------------------------------------------------

    top_segments = [
        (
            project(x_left, depth_start, height_top),
            project(x_right, depth_start, height_top)
        ),
        (
            project(x_right, depth_start, height_top),
            project(x_right, depth_end, height_top)
        ),
    ]

    for segment_start, segment_end in top_segments:
        ax.plot(
            [segment_start[0], segment_end[0]],
            [segment_start[1], segment_end[1]],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            zorder=45 + step_index
        )

    # --------------------------------------------------------
    # BORD MODIFIÉ : liaison vers la base du bord gauche
    # de la façade SAT de la marche suivante
    # --------------------------------------------------------
    #
    # Le point de départ reste le point gauche de la marche
    # actuelle, au début de sa profondeur.
    #
    # Le point d'arrivée devient le pied du bord gauche
    # de la façade de la marche suivante.
    #
    # La dernière marche n'a pas de marche suivante.
    # --------------------------------------------------------
    if step_index < N_STEPS - 1:

        next_start_year = STEP_WINDOWS[
            step_index + 1
        ][0]

        next_x_left = step_left(
            step_index + 1
        )

        left_edge_start = project(
            x_left,
            depth_start,
            height_top
        )

        left_edge_end = project(
            next_x_left,
            next_start_year,
            height_top
        )

        ax.plot(
            [left_edge_start[0], left_edge_end[0]],
            [left_edge_start[1], left_edge_end[1]],
            color="black",
            linewidth=EDGE_LW,
            zorder=60 + step_index
        )


    # ========================================================
    # 18b. FAÇADE : SAT
    # ========================================================

    for annual_index, year in enumerate(
        range(
            start_year,
            end_year + 1
        )
    ):

        fraction0 = (
            annual_index / 30.0
        )

        fraction1 = (
            (annual_index + 1) / 30.0
        )


        x0 = (
            x_left
            + fraction0 * width
        )

        x1 = (
            x_left
            + fraction1 * width
        )


        points = [

            project(
                x0,
                depth_start,
                height_bottom
            ),

            project(
                x1,
                depth_start,
                height_bottom
            ),

            project(
                x1,
                depth_start,
                height_top
            ),

            project(
                x0,
                depth_start,
                height_top
            )

        ]


        value = value_for_year(
            dT,
            year
        )


        # ====================================================
        # SAT COMPLÈTEMENT OPAQUE
        # ====================================================

        add_polygon(
            ax,
            points,

            facecolor=color_for_value(
                CMAP_T,
                norm_T,
                value
            ),

            edgecolor="none",

            linewidth=0,

            # aucune transparence
            alpha=1.0,

            # évite les petits traits transparents
            # entre les bandes
            antialiased=False,

            zorder=25 + step_index
        )


    # --------------------------------------------------------
    # Contour de la façade
    # --------------------------------------------------------

    facade_outline = [

        project(
            x_left,
            depth_start,
            height_bottom
        ),

        project(
            x_right,
            depth_start,
            height_bottom
        ),

        project(
            x_right,
            depth_start,
            height_top
        ),

        project(
            x_left,
            depth_start,
            height_top
        )

    ]


    add_polygon(
        ax,
        facade_outline,
        facecolor="none",
        edgecolor=EDGE_COLOR,
        linewidth=EDGE_LW,
        zorder=55 + step_index
    )


# ============================================================
# 19. FLANC DROIT PET
# ============================================================
#
# IMPORTANT :
# Le bord droit reste BASE_STEP_WIDTH pour toutes les marches.
# Ainsi le flanc reste un plan géométrique continu,
# malgré la réduction progressive de la largeur.

X_PET = BASE_STEP_WIDTH


for year in range(
    OBS_START,
    END_YEAR + 1
):

    year_next = (
        year + 1
    )

    height = stair_height(
        year
    )


    points = [

        project(
            X_PET,
            year,
            0.0
        ),

        project(
            X_PET,
            year_next,
            0.0
        ),

        project(
            X_PET,
            year_next,
            height
        ),

        project(
            X_PET,
            year,
            height
        )

    ]


    value = value_for_year(
        dPET,
        year
    )


    add_polygon(
        ax,
        points,

        facecolor=color_for_value(
            CMAP_PET,
            norm_PET,
            value
        ),

        edgecolor="none",
        linewidth=0,

        alpha=1.0,
        antialiased=False,

        zorder=80
    )


# ============================================================
# 20. CONTOURS DU FLANC PET
# ============================================================

# Base diagonale
base_start = project(
    X_PET,
    OBS_START,
    0.0
)

base_end = project(
    X_PET,
    END_YEAR + 1,
    0.0
)


ax.plot(
    [
        base_start[0],
        base_end[0]
    ],
    [
        base_start[1],
        base_end[1]
    ],
    color=EDGE_COLOR,
    linewidth=0.9,
    zorder=110
)


# Bord vertical avant
initial_top = project(
    X_PET,
    OBS_START,
    STEP_HEIGHT
)


ax.plot(
    [
        base_start[0],
        initial_top[0]
    ],
    [
        base_start[1],
        initial_top[1]
    ],
    color=EDGE_COLOR,
    linewidth=0.9,
    zorder=110
)


# ============================================================
# 21. PROFIL SUPÉRIEUR EN ESCALIER
# ============================================================

profile_points = []


for step_index, start_year in enumerate(
    STEP_STARTS
):

    height = (
        (step_index + 1)
        * STEP_HEIGHT
    )


    if step_index < N_STEPS - 1:

        end_profile = (
            STEP_STARTS[
                step_index + 1
            ]
        )

    else:

        end_profile = (
            END_YEAR + 1
        )


    if step_index == 0:

        profile_points.append(
            project(
                X_PET,
                start_year,
                height
            )
        )


    profile_points.append(
        project(
            X_PET,
            end_profile,
            height
        )
    )


    # Montée verticale
    if step_index < N_STEPS - 1:

        profile_points.append(
            project(
                X_PET,
                end_profile,
                height + STEP_HEIGHT
            )
        )


profile_points = np.asarray(
    profile_points
)


ax.plot(
    profile_points[:, 0],
    profile_points[:, 1],
    color=EDGE_COLOR,
    linewidth=0.9,
    zorder=111
)


# ============================================================
# 22. BORD TERMINAL
# ============================================================

end_bottom = project(
    X_PET,
    END_YEAR + 1,
    0.0
)

end_top = project(
    X_PET,
    END_YEAR + 1,
    N_STEPS * STEP_HEIGHT
)


ax.plot(
    [
        end_bottom[0],
        end_top[0]
    ],
    [
        end_bottom[1],
        end_top[1]
    ],
    color=EDGE_COLOR,
    linewidth=0.9,
    zorder=111
)


# ============================================================
# 23. ÉTIQUETTES DES MARCHES
# ============================================================

for step_index, (
    start_year,
    end_year
) in enumerate(
    STEP_WINDOWS
):

    height = (
        (step_index + 1)
        * STEP_HEIGHT
    )

    x_text = (
        step_left(step_index)
        - 0.20
    )

    position = project(
        x_text,
        start_year,
        height - 0.08
    )


    ax.text(
        position[0] - 0.08,
        position[1],

        f"{start_year} – {end_year}",

        ha="right",
        va="center",

        fontsize=9.2,

        color="black",

        zorder=150
    )


# ============================================================
# 24. ANNÉES SUR L'AXE PET
# ============================================================

for year in [
    1901,
    1950,
    2000,
    2035
]:

    p = project(
        X_PET,
        year,
        0.0
    )


    ax.text(
        p[0],
        p[1] - 0.34,

        str(year),

        ha="center",
        va="top",

        fontsize=9,

        zorder=150
    )


# ============================================================
# 25. TITRES
# ============================================================

ax.text(
    0.0,
    0.975,
    "Stairway to even more extreme weather events",
    transform=ax.transAxes,
    fontsize=25,
    fontweight="bold",
    ha="left",
    va="bottom"
)

ax.text(
    0.0,
    0.912,
    "30-year stairs with a 15-year overlapping",
    transform=ax.transAxes,
    fontsize=12.5,
    ha="left",
    va="bottom"
)


# ============================================================
# 26. LIMITES
# ============================================================

geometry_points = []


geometry_points.extend([

    project(
        X_PET,
        OBS_START,
        0.0
    ),

    project(
        X_PET,
        END_YEAR + 1,
        0.0
    )

])


for step_index, start_year in enumerate(
    STEP_STARTS
):

    height = (
        (step_index + 1)
        * STEP_HEIGHT
    )


    if step_index < N_STEPS - 1:

        end_year = (
            STEP_STARTS[
                step_index + 1
            ]
        )

    else:

        end_year = (
            END_YEAR + 1
        )


    x_left = step_left(
        step_index
    )

    x_right = step_right(
        step_index
    )


    geometry_points.extend([

        project(
            x_left,
            start_year,
            height
        ),

        project(
            x_right,
            start_year,
            height
        ),

        project(
            x_left,
            end_year,
            height
        ),

        project(
            x_right,
            end_year,
            height
        )

    ])


geometry_points = np.asarray(
    geometry_points
)


xmin = np.min(
    geometry_points[:, 0]
)

xmax = np.max(
    geometry_points[:, 0]
)

ymin = np.min(
    geometry_points[:, 1]
)

ymax = np.max(
    geometry_points[:, 1]
)


dx = xmax - xmin
dy = ymax - ymin


ax.set_xlim(
    xmin - 0.16 * dx,
    xmax + 0.05 * dx
)

ax.set_ylim(
    ymin - 0.10 * dy,
    ymax + 0.08 * dy
)


# ============================================================
# 27. COLORBARS
# ============================================================

CB_Y = 0.145
CB_H = 0.025
CB_W = 0.235

CB_X1 = 0.125
CB_X2 = 0.402
CB_X3 = 0.679


sm_T = mpl.cm.ScalarMappable(
    norm=norm_T,
    cmap=CMAP_T
)

sm_P = mpl.cm.ScalarMappable(
    norm=norm_P,
    cmap=CMAP_P
)

sm_PET = mpl.cm.ScalarMappable(
    norm=norm_PET,
    cmap=CMAP_PET
)


cax_T = fig.add_axes([
    CB_X1,
    CB_Y,
    CB_W,
    CB_H
])

cax_P = fig.add_axes([
    CB_X2,
    CB_Y,
    CB_W,
    CB_H
])

cax_PET = fig.add_axes([
    CB_X3,
    CB_Y,
    CB_W,
    CB_H
])


cb_T = fig.colorbar(
    sm_T,
    cax=cax_T,
    orientation="horizontal"
)

cb_P = fig.colorbar(
    sm_P,
    cax=cax_P,
    orientation="horizontal"
)

cb_PET = fig.colorbar(
    sm_PET,
    cax=cax_PET,
    orientation="horizontal"
)


cb_T.ax.set_title(
    "SAT anomalies (K)",
    fontsize=11,
    pad=7,
#   fontweight="bold"
)

cb_P.ax.set_title(
    "Precipitation anomalies (mm/day)",
    fontsize=11,
    pad=7,
#   fontweight="bold"
)

cb_PET.ax.set_title(
    "ETP anomalies (mm/day)",
    fontsize=11,
    pad=7,
#   fontweight="bold"
)


for cb in [
    cb_T,
    cb_P,
    cb_PET
]:

    cb.ax.tick_params(
        labelsize=8,
        length=3
    )


# ============================================================
# 28. ANNOTATIONS SOUS LES BARRES DE COULEUR
# ============================================================

ANNOT_X1 = 0.31
ANNOT_X2 = 0.60
ANNOT_X3 = 0.86
ANNOT_Y = 0.10

fig.text(
    ANNOT_X1, ANNOT_Y,
    "More heatwaves",
    ha="center",
    va="center",
    fontsize=11
)

fig.text(
    ANNOT_X2, ANNOT_Y,
    "More floods",
    ha="center",
    va="center",
    fontsize=11
)

fig.text(
    ANNOT_X3, ANNOT_Y,
    "More droughts",
    ha="center",
    va="center",
    fontsize=11
)
# ============================================================
# 29. EXPORT
# ============================================================

plt.savefig(
    "Fig7.png",
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Fig7.pdf",
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ============================================================
# 30. FERMETURE
# ============================================================

ds_t.close()
ds_p.close()
ds_pet.close()
