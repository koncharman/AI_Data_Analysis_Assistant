from typing import Any, Dict, Optional

import pandas as pd


class DatasetContext:
    """Store the active dataset for agent-facing analysis tools."""

    def __init__(self) -> None:
        self._dataframe: Optional[pd.DataFrame] = None
        self._profile: Optional[Dict[str, Any]] = None
        self._dataset_id: Optional[str] = None

    def set_dataset(
        self,
        dataframe: pd.DataFrame,
        profile: Dict[str, Any],
        dataset_id: Optional[str] = None,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if not isinstance(profile, dict):
            raise TypeError("profile must be a dictionary.")
        self._dataframe = dataframe
        self._profile = profile
        self._dataset_id = dataset_id

    def clear(self) -> None:
        self._dataframe = None
        self._profile = None
        self._dataset_id = None

    def is_ready(self) -> bool:
        return self._dataframe is not None and self._profile is not None

    def get_dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("No active dataset is available.")
        return self._dataframe

    def get_profile(self) -> Dict[str, Any]:
        if self._profile is None:
            raise RuntimeError("No active dataset profile is available.")
        return self._profile

    def get_dataset_id(self) -> Optional[str]:
        return self._dataset_id


dataset_context = DatasetContext()
