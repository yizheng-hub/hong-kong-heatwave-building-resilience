# Intermediate data

`intermediate/source_derived/` contains generated source-adjacent mappings and prepared weather products. `intermediate/prepared_stock/` contains stable building-to-component and spatial mappings used by several figures. `intermediate/spatial/` contains prepared boundaries and simplified roads used for map rendering. `intermediate/city_trajectories/` contains an accepted 15-minute city trajectory used for the 2009 peak-definition diagnostic.

The complete local root additionally retains `compiled_simulation_inputs/` and `simulation_outputs/`. These are the frozen physical inputs and accepted solver records needed to audit or rerun the production chain; they are excluded from Git because of size.

These files are machine-readable products between primary data preparation/simulation and the final analysis tables. They are included because they are needed by the executable notebook and are substantially smaller than the original simulator output hierarchy.
