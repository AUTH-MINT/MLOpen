from django.db import models


class InputFile(models.Model):
    """
    Model for file uploads
    """
    name = models.CharField(max_length=200, db_index=True)
    created_at = models.DateTimeField(null=True)
    file = models.FileField()

    def __str__(self):
        return self.name
