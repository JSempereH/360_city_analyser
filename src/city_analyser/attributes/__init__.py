"""Structural and descriptive building attributes, kept apart from footprint association.

Attribute providers consume ranked crops (``building_inspection.crops``) or OSM
tags and return per-attribute predictions with explicit abstentions. They never
touch the footprint assessment or reuse its scores (see ARCHITECTURE.md).
"""
