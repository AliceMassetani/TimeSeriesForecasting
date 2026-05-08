from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ..entities.backtest_result import BacktestResult
from ..schemas.backtest_chart import BacktestChart

class IBacktestRepository(ABC):
    """
    Interfaccia astratta per il Repository del Backtest.
    """
    
    @abstractmethod
    def get_chart_data(self, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BacktestChart:
        pass

    @abstractmethod
    def create_bulk(self, results: List[BacktestResult]) -> int:
        pass

    @abstractmethod
    def get_all(self) -> List[BacktestResult]:
        pass

    @abstractmethod
    def delete_all(self) -> int:
        pass

class IForecastRepository(ABC):
    """
    Interfaccia astratta per il Repository del Forecast futuro.
    """
    @abstractmethod
    def get_chart_data(self, limit: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> "ForecastChart":
        pass

    @abstractmethod
    def create_bulk(self, results: List[any]) -> int:
        pass

    @abstractmethod
    def delete_all(self) -> int:
        pass
