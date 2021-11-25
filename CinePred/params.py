# IMPORT
import numpy as np
import pandas as pd
from currency_converter import CurrencyConverter
from xgboost import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from CinePred.data.utils import *
from CinePred.data.data import Data
from CinePred.data.genre_ohe import GenreOHE
