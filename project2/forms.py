from django import forms

from .services.dataset import NUMERIC_FEATURES


class CounterfactualForm(forms.Form):
    row_index = forms.IntegerField(min_value=0)
    target_class = forms.CharField()
    model_type = forms.CharField()
    lambda_value = forms.FloatField(min_value=0)


class EffectForm(forms.Form):
    feature = forms.ChoiceField(choices=[(f, f) for f in NUMERIC_FEATURES])
    model_type = forms.CharField()
    lambda_value = forms.FloatField(min_value=0)
