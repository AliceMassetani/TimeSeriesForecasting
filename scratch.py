from darts.models import TSMixerModel
import inspect

print(inspect.signature(TSMixerModel.gridsearch))
print(TSMixerModel.gridsearch.__doc__)
