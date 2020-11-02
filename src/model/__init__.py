from src.model._base import BaseNN
from src.model._dropout import DropoutNN
from src.model._maxout import MaxoutNN
from src.model._maxout_with_dropout import MaxoutWithDropoutNN

model_map = {
    'base': BaseNN,
    'dropout': DropoutNN,
    'maxout': MaxoutNN,
    'maxout_with_dropout': MaxoutWithDropoutNN,
}
