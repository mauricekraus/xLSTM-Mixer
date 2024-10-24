from enum import Enum


class Task(Enum):
    LONG_TERM_FORECAST = "long_term_forecast"
    SHORT_TERM_FORECAST = "short_term_forecast"
    IMPUTATION = "imputation"
    CLASSIFICATION = "classification"

    @property
    def is_shuffable(self):
        return self not in [Task.LONG_TERM_FORECAST, Task.SHORT_TERM_FORECAST, Task.IMPUTATION]


class ForecastingTaskOptions(Enum):
    MULTIVARIATE_2_MULTIVARIATE = "M"
    MULTIVARIATE_2_UNIVARIATE = "MS"
    UNIVARIATE_2_UNIVARIATE = "S"


class TimeFreq(Enum):
    SECONDLY = "s"
    MINUTELY = "t"
    HOURLY = "h"
    DAILY = "d"
    BUSSINESS_DAYS = "b"
    WEEKLY = "w"
    MONTHLY = "m"
    FIFTEEN_MINUTES = "15min"
    THREE_HOURS = "3h"