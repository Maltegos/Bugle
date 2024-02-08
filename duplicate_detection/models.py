from django.db import models


# Create your models here.

class BugReport(models.Model):
    # TODO add summary as well
    description = models.TextField(max_length=2000)

    def __str__(self):
        return self.description
