# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2018-01-20 18:17
from __future__ import unicode_literals

import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dvaapp', '0002_trainedmodel_training_set'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingset',
            name='built_ts',
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name='trainingset',
            name='source_filters',
            field=django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='trainingset',
            name='event',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent'),
        ),
        migrations.AlterField(
            model_name='trainingset',
            name='instance_type',
            field=models.CharField(choices=[('I', 'images'), ('X', 'index'), ('V', 'videos')], db_index=True, default='I', max_length=1),
        ),
        migrations.AlterField(
            model_name='trainingset',
            name='training_task_type',
            field=models.CharField(choices=[('D', 'Detection'), ('I', 'Indexing'), ('A', 'LOPQ Approximation'), ('C', 'Classification')], db_index=True, default='D', max_length=1),
        ),
    ]
