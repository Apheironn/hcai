from django import forms

AGE_GROUPS = [("18-24", "18-24"), ("25-34", "25-34"), ("35-49", "35-49"), ("50+", "50 or older")]

MOVIE_FREQUENCY = [
    ("weekly", "At least once a week"),
    ("monthly", "A few times a month"),
    ("rarely", "Less than once a month"),
]

LIKERT = [(str(i), label) for i, label in enumerate(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"], start=1)]


class StartStudyForm(forms.Form):
    participant_code = forms.CharField(max_length=20, label="Participant code")
    age_group = forms.ChoiceField(choices=AGE_GROUPS, label="Age group")
    movie_frequency = forms.ChoiceField(choices=MOVIE_FREQUENCY, label="How often do you watch movies?")
    consent = forms.BooleanField(label="I have read the information above and agree to take part.")


class SurveyForm(forms.Form):
    effort = forms.ChoiceField(choices=LIKERT, widget=forms.RadioSelect, label="This task was mentally demanding.")
    ease = forms.ChoiceField(choices=LIKERT, widget=forms.RadioSelect, label="It was easy to express my preferences.")
    confidence = forms.ChoiceField(
        choices=LIKERT, widget=forms.RadioSelect, label="I am confident the system understood my taste."
    )
