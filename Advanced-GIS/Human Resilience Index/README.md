
## Human Resilience Index — A Composite Spatial Indicator for Global Socioeconomic Analysis

Building a multi-dimensional resilience index from income, gender parity, and education, then mapping it globally in both GeoPandas and ArcGIS Pro.

## Overview

Financial wealth is a poor proxy for a society's ability to absorb a crisis. A country with high GDP but severe gender inequality and low educational attainment is more fragile in the face of economic shocks than its income suggests. Conversely, a poorer country with strong education and equitable institutions may be far more resilient than its raw economic output would indicate.

This project builds a composite index — the Human Resilience Index (HRI) — that combines three observable indicators into a single score, then maps it globally at country resolution. The index is deliberately transparent: the weighting scheme is stated, the normalization is documented, and the two independent implementations (Python and ArcGIS Pro) should produce matching results.

## Index Construction

The composite index is defined as:

```
HRI = EI × 0.2 + (1 − GII) × 0.3 + GNI' × 0.5
```

Three components:

- **EI** — Education Index. Higher values indicate better educational attainment.
- **GII** — Gender Inequality Index. Lower values indicate more equality. The `(1 − GII)` transform flips the direction so that higher values are better.
- **GNI'** — Gross National Income, log-normalized. See below.

### Why log-normalize GNI

Raw GNI spans three orders of magnitude across countries. A linear normalization would compress most countries into a narrow band at the low end of the scale and let a handful of high-income countries dominate the upper range. The result would be a map that essentially shows "rich vs poor" and ignores the middle of the distribution.

Log-normalization expands the middle:

```
GNI' = (log(GNI) − min(log(GNI))) / (max(log(GNI)) − min(log(GNI)))
```

After this transform, the difference between a $2,000 and $10,000 economy is on the same scale as the difference between a $40,000 and $200,000 economy. Both matter proportionally to their log ratio, which is closer to how the underlying social outcomes actually behave — the marginal improvement in resilience from moving out of poverty is larger than the marginal improvement from moving from "comfortable" to "very comfortable".

### Weighting rationale

The weights were chosen as follows:

- **GNI (50%)** — economic capacity remains the strongest single predictor of crisis absorption
- **Gender equality (30%)** — societies with poor gender parity tend to have systematically weaker institutional resilience
- **Education (20%)** — an important but partially downstream factor; education outcomes are partly explained by the other two

There is no objective "right" weighting scheme. Any composite index embeds a value judgment about which components matter most. The point of publishing the weights explicitly is that users can disagree with the judgment, apply different weights, and see whether the resulting map changes.

## Two Implementations

The index was built twice, independently, to cross-validate the pipeline.

### Implementation 1 — Python with GeoPandas

Data was loaded as a shapefile with country boundaries and joined to a table of EI, GII, and GNI values by ISO country code. The three components were computed, normalized, and combined.

```python
import geopandas as gpd
import numpy as np

# Load and join
world = gpd.read_file('world_countries.shp')
indicators = pd.read_csv('indicators.csv')
merged = world.merge(indicators, left_on='ISO_A3', right_on='ISO')

# Normalize GNI logarithmically
log_gni = np.log(merged['GNI'])
gni_norm = (log_gni - log_gni.min()) / (log_gni.max() - log_gni.min())

# Composite index
merged['HRI'] = (
    merged['EI'] * 0.2 +
    (1 - merged['GII']) * 0.3 +
    gni_norm * 0.5
)

# Plot
merged.plot(
    column='HRI',
    cmap='Greens',
    scheme='natural_breaks',
    k=5,
    legend=True,
    edgecolor='white',
    linewidth=0.3
)
```

The GeoPandas plot uses the same Natural Breaks (Jenks) classification as the ArcGIS version. This is intentional — a fair comparison between the two implementations requires both to use the same classification scheme.

![Human Resilience Index — global choropleth rendered in Python with GeoPandas using Natural Breaks classification](../../assets/Images/Projects/Advanced%20GIS/HRI%20Final%20Map%20(Python).png)

### Implementation 2 — ArcGIS Pro

The same data, joined to the same attribute table, symbolised with the same Jenks classification (5 classes) and the same green continuous color ramp. The ArcGIS version has full cartographic furniture: title, scale bar, north arrow, graticule, legend, source credits, and an inset overview.

The two maps are visually indistinguishable at the classification level. Every country falls into the same class in both versions.

![Human Resilience Index — the same global choropleth re-rendered in ArcGIS Pro with full cartographic furniture](../../assets/Images/Projects/Advanced%20GIS/HRI%20Final%20Map%20(ArcGIS%20Pro).png)

## Map Design

The final map is a **world choropleth** at global scale. Design decisions:

**Classification — Natural Breaks (Jenks), 5 classes.** Jenks minimizes within-class variance and maximizes between-class variance. For a distribution with clusters like the HRI, this produces classes that are actually distinguishable on the map, rather than arbitrary equal-interval breaks that ignore the data structure.

**Color scheme — sequential green, 5 steps.** Green was chosen over red-to-blue diverging schemes because the index has a natural direction: more resilient is better. A diverging scheme would imply a meaningful center point that the index does not have.

**Legend.** Discrete classes with explicit numerical ranges: `0.21–0.36`, `0.36–0.52`, `0.52–0.67`, `0.67–0.83`, `0.83–1.00`. Discrete ranges in the legend, continuous gradient on the map, is the correct pairing for a Jenks-classified choropleth.

**Gestalt principle — Figure and Ground.** The land areas are the figure; the ocean is the ground. This was achieved by keeping oceans as light neutral (or transparent) and giving land polygons strong borders. The visual hierarchy makes countries read as objects rather than as part of a uniform field.

**Standard cartographic elements** included:

- Title: "Human Resilience Index (HRI)"
- Subtitle: classification method and year
- Legend with 5 discrete classes plus a "No Data" swatch
- Scale bar (in km)
- North arrow
- Graticule (latitude/longitude grid)
- Source credits: "Source: WorldPopulationReview.com"
- Cartographer credit: "Cartography and Index: Ali Rezaali"

## What the Map Shows

Four regional patterns are visible:

**Northern Europe, North America, Australia, and Japan** occupy the highest two classes. These are the expected high-resilience economies — high income, strong gender parity, and well-developed education systems all pull in the same direction.

**Sub-Saharan Africa and parts of South Asia** occupy the lowest two classes. This is also expected, and the map makes the pattern visible at a glance: resilience deficits cluster geographically.

**Latin America and Eastern Europe** span the middle three classes with significant internal variation. Chile, Uruguay, and Costa Rica score notably higher than their regional neighbors; Russia and several Central Asian states score lower than their income levels alone would predict, because gender parity pulls the composite down.

**Middle Eastern petro-states** show an interesting pattern — high GNI does not always translate to high HRI. Several countries in the region that rank very high on income alone fall into the middle classes of the composite, precisely because the index penalizes low gender parity and uneven educational attainment. This is the pattern the composite index was designed to expose.

## Cross-Validation Between the Two Implementations

The most interesting technical outcome of the project is the cross-check. The two implementations used different libraries (GeoPandas vs. ArcGIS Pro's rendering engine), different data joins (Python merges vs. ArcGIS's table join), and different plotting paths (matplotlib backend vs. ArcGIS rendering). Yet they produced maps that agreed class-for-class across every country.

This is worth more than it looks. It means:

1. The data pipeline is reproducible across platforms.
2. The classification algorithm produces identical results when the input distribution is identical.
3. The composite index itself is platform-independent — it depends only on the input data and the weights, not on the software.

A user replicating this analysis in QGIS, R, or plain JavaScript should get the same map. That is a property worth verifying explicitly, because it establishes the analysis as *scientific* rather than *software-specific*.