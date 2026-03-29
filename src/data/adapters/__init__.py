# -*- coding: utf-8 -*-
"""
src/data/adapters/__init__.py
"""
from src.data.adapters.base_adapter import BaseAdapter
from src.data.adapters.registry import get_adapter
from src.data.adapters.coverage import generate_coverage_report

__all__ = ["BaseAdapter", "get_adapter", "generate_coverage_report"]
