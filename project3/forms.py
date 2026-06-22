from django import forms


class ActiveLearningForm(forms.Form):
    budget = forms.IntegerField(min_value=100, max_value=5000, initial=2000)
    batch_size = forms.IntegerField(min_value=50, max_value=500, initial=100)
