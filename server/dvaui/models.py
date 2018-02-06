# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import subprocess, os, json, logging, tempfile, shutil
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField
from django.utils.translation import ugettext_lazy as _


class ExternalServer(models.Model):
    name = models.CharField(max_length=300,default="",unique=True)
    url = models.CharField(max_length=1000,default="",unique=True)
    created = models.DateTimeField('date created', auto_now_add=True)

    class Meta:
        unique_together = (("name", "url"),)

    def pull(self):
        errors = []
        cwd = tempfile.mkdtemp()
        gitpull = subprocess.Popen(['git', 'clone', self.url, self.name], cwd=cwd)
        gitpull.wait()
        for root, directories, filenames in os.walk("{}/{}/".format(cwd,self.name)):
            for filename in filenames:
                if filename.endswith('.json'):
                    try:
                        j = json.load(file(os.path.join(root,filename)))
                    except:
                        errors.append(filename)
                    else:
                        url = self.url
                        if url.endswith('/'):
                            url = url[:-1]
                        relpath = os.path.join(root,filename)[len(cwd)+1+len(self.name):]
                        if relpath.startswith('/'):
                            relpath = relpath[1:]
                        flname = "{}/{}".format(url,relpath)
                        p, _ = StoredDVAPQL.objects.get_or_create(name=flname,server=self)
                        p.server = self
                        p.process_type = StoredDVAPQL.PROCESS
                        p.script = j
                        p.description = j.get('description',"")
                        p.save()
        shutil.rmtree(cwd)
        if errors:
            logging.warning("Could not import {}".format(errors))
        return errors


class StoredDVAPQL(models.Model):
    """
    Stored processes
    """
    SCHEDULE = 'S'
    PROCESS = 'V'
    QUERY = 'Q'
    TYPE_CHOICES = ((SCHEDULE, '执行计划'), (PROCESS, '进程'), (QUERY, '查询'))
    process_type = models.CharField(max_length=1, choices=TYPE_CHOICES, default=QUERY,db_index=True, verbose_name="处理类型")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True, )
    creator = models.ForeignKey(User, null=True, related_name="script_creator", verbose_name="创建人")
    name = models.CharField(max_length=300,default="", verbose_name="名称")
    description = models.TextField(blank=True,default="", verbose_name="描述")
    server = models.ForeignKey(ExternalServer,null=True, verbose_name="服务器")
    script = JSONField(blank=True, null=True, verbose_name="脚本")
    class Meta:
        verbose_name = _("Stored dvapqls")
        verbose_name_plural = _("Stored dvapqls")




