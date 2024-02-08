from .models import BugReport
from django import forms
from django.forms import ModelForm


class BugReportForm(ModelForm):
    description = forms.TextInput()

    class Meta:
        model = BugReport
        fields = ('description',)

