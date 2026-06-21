from django import forms


class UploadForm(forms.Form):
    file = forms.FileField(label="CSV file")


class PlotForm(forms.Form):
    problem_type = forms.ChoiceField(
        choices=[
            ("auto", "Auto"),
            ("classification", "Classification"),
            ("regression", "Regression"),
        ],
        initial="auto",
    )
    x_feature = forms.CharField(required=False)
    y_feature = forms.CharField(required=False)


class TrainForm(forms.Form):
    model = forms.CharField()
    test_size = forms.IntegerField(min_value=10, max_value=90, initial=80)
    param_values = forms.CharField(required=False)
    metric = forms.CharField()
