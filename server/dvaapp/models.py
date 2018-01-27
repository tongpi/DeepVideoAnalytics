# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os, json, gzip, sys, shutil, zipfile, uuid
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField
from django.conf import settings
from django.utils import timezone
from . import fs
try:
    import numpy as np
except ImportError:
    pass
from uuid import UUID
from json import JSONEncoder
JSONEncoder_old = JSONEncoder.default


def JSONEncoder_new(self, o):
    if isinstance(o, UUID): return str(o)
    return JSONEncoder_old(self, o)


JSONEncoder.default = JSONEncoder_new


class Worker(models.Model):
    queue_name = models.CharField(max_length=500, default="", verbose_name="队列名称")
    host = models.CharField(max_length=500, default="", verbose_name="主机")
    pid = models.IntegerField()
    alive = models.BooleanField(default=True, verbose_name="激活")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)


class DVAPQL(models.Model):
    SCHEDULE = 'S'
    PROCESS = 'V'
    QUERY = 'Q'
    TYPE_CHOICES = ((SCHEDULE, '执行计划'), (PROCESS, '进程'), (QUERY, '查询'))
    process_type = models.CharField(max_length=1, choices=TYPE_CHOICES, default=QUERY, verbose_name="处理类型")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    user = models.ForeignKey(User, null=True, related_name="submitter", verbose_name="用户")
    script = JSONField(blank=True, null=True, verbose_name="脚本")
    results_metadata = models.TextField(default="", verbose_name="元数据结果集")
    results_available = models.BooleanField(default=False, verbose_name="结果集有效")
    completed = models.BooleanField(default=False, verbose_name="已完成")
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)


class Video(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    name = models.CharField(max_length=500,default="", verbose_name="名称")
    length_in_seconds = models.IntegerField(default=0, verbose_name="时长（秒）")
    height = models.IntegerField(default=0, verbose_name="高度")
    width = models.IntegerField(default=0, verbose_name="宽度")
    metadata = models.TextField(default="", verbose_name="元数据")
    frames = models.IntegerField(default=0, verbose_name="帧数量")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    description = models.TextField(default="", verbose_name="描述")
    uploaded = models.BooleanField(default=False, verbose_name="已上传")
    dataset = models.BooleanField(default=False, verbose_name="数据集")
    uploader = models.ForeignKey(User,null=True, verbose_name="上传者")
    segments = models.IntegerField(default=0, verbose_name="段数量")
    url = models.TextField(default="", verbose_name="网址")
    youtube_video = models.BooleanField(default=False, verbose_name="youtube视频")
    parent_process = models.ForeignKey(DVAPQL,null=True, verbose_name="父进程")

    def __unicode__(self):
        return u'{}'.format(self.name)

    def path(self,media_root=None):
        if not (media_root is None):
            return "{}/{}/video/{}.mp4".format(media_root, self.pk, self.pk)
        else:
            return "{}/{}/video/{}.mp4".format(settings.MEDIA_ROOT,self.pk,self.pk)

    def get_frame_list(self,media_root=None):
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        framelist_path = "{}/{}/framelist".format(media_root, self.pk)
        if os.path.isfile('{}.json'.format(framelist_path)):
            return json.load(file('{}.json'.format(framelist_path)))
        elif os.path.isfile('{}.gz'.format(framelist_path)):
            return json.load(gzip.GzipFile('{}.gz'.format(framelist_path)))
        else:
            raise ValueError("Frame list could not be found at {}".format(framelist_path))

    def create_directory(self, create_subdirs=True):
        d = '{}/{}'.format(settings.MEDIA_ROOT, self.pk)
        if not os.path.exists(d):
            try:
                os.mkdir(d)
            except OSError:
                pass
        if create_subdirs:
            for s in ['video','frames','segments','indexes','regions','transforms','audio']:
                d = '{}/{}/{}/'.format(settings.MEDIA_ROOT, self.pk, s)
                if not os.path.exists(d):
                    try:
                        os.mkdir(d)
                    except OSError:
                        pass


class IngestEntry(models.Model):
    video = models.ForeignKey(Video, verbose_name="视频")
    ingest_index = models.IntegerField()
    ingest_filename = models.CharField(max_length=500)
    start_segment_index = models.IntegerField(null=True, verbose_name="开始段索引")
    start_frame_index = models.IntegerField(null=True, verbose_name="开始帧索引")
    segments = models.IntegerField(null=True, verbose_name="段")
    frames = models.IntegerField(null=True, verbose_name="帧")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)

    class Meta:
        unique_together = (("video", "ingest_filename","ingest_index"),)


class TEvent(models.Model):
    started = models.BooleanField(default=False, verbose_name="已启动")
    completed = models.BooleanField(default=False, verbose_name="已完成")
    errored = models.BooleanField(default=False, verbose_name="已出错")
    worker = models.ForeignKey(Worker, null=True, verbose_name="工作进程")
    error_message = models.TextField(default="", verbose_name="错误信息")
    video = models.ForeignKey(Video, null=True, verbose_name="视频")
    operation = models.CharField(max_length=100, default="", verbose_name="操作")
    queue = models.CharField(max_length=100, default="", verbose_name="队列名称")
    created = models.DateTimeField(verbose_name="已创建时长", auto_now_add=True)
    start_ts = models.DateTimeField(verbose_name="启动时间", null=True)
    duration = models.FloatField(default=-1, verbose_name="运行时长")
    arguments = JSONField(blank=True,null=True, verbose_name="算法")
    task_id = models.TextField(null=True, verbose_name="任务ID")
    parent = models.ForeignKey('self',null=True)
    parent_process = models.ForeignKey(DVAPQL,null=True, verbose_name="父进程")
    imported = models.BooleanField(default=False, verbose_name="已导入")
    task_group_id = models.IntegerField(default=-1)


class TrainedModel(models.Model):
    """
    A model Model
    """
    TENSORFLOW = 'T'
    CAFFE = 'C'
    PYTORCH = 'P'
    OPENCV = 'O'
    MXNET = 'M'
    MODES = (
        (TENSORFLOW, 'Tensorflow'),
        (CAFFE, 'Caffe'),
        (PYTORCH, 'Pytorch'),
        (OPENCV, 'OpenCV'),
        (MXNET, 'MXNet'),
    )
    INDEXER = 'I'
    APPROXIMATOR = 'P'
    DETECTOR = 'D'
    ANALYZER = 'A'
    SEGMENTER = 'S'
    MTYPE = (
        (APPROXIMATOR, 'Approximator'),
        (INDEXER, 'Indexer'),
        (DETECTOR, 'Detector'),
        (ANALYZER, 'Analyzer'),
        (SEGMENTER, 'Segmenter'),
    )
    YOLO = "Y"
    TFD = "T"
    DETECTOR_TYPES = (
        (TFD, 'Tensorflow'),
        (YOLO, 'YOLO V2'),
    )
    detector_type = models.CharField(max_length=1,choices=DETECTOR_TYPES,db_index=True,null=True)
    mode = models.CharField(max_length=1,choices=MODES,db_index=True,default=TENSORFLOW)
    model_type = models.CharField(max_length=1,choices=MTYPE,db_index=True,default=INDEXER)
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    shasum = models.CharField(max_length=40,null=True,unique=True)
    model_filename = models.CharField(max_length=200,default="",null=True)
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    arguments = JSONField(null=True,blank=True)
    source = models.ForeignKey(TEvent, null=True)
    trained = models.BooleanField(default=False)
    url = models.CharField(max_length=200,default="")
    files = JSONField(null=True,blank=True)
    produces_labels = models.BooleanField(default=False)
    produces_json = models.BooleanField(default=False)
    produces_text = models.BooleanField(default=False)
    # Following allows us to have a hierarchy of models (E.g. inception pretrained -> inception fine tuned)
    parent = models.ForeignKey('self', null=True)
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    def create_directory(self,create_subdirs=True):
        try:
            os.mkdir('{}/models/{}'.format(settings.MEDIA_ROOT, self.uuid))
        except:
            pass

    def get_model_path(self,root_dir=None):
        if root_dir is None:
            root_dir = settings.MEDIA_ROOT
        if self.model_filename:
            return "{}/models/{}/{}".format(root_dir,self.uuid,self.model_filename)
        elif self.files:
            return "{}/models/{}/{}".format(root_dir,self.uuid, self.files[0]['filename'])
        else:
            return None

    def get_yolo_args(self):
        model_dir = "{}/models/{}/".format(settings.MEDIA_ROOT, self.uuid)
        class_names = {k: v for k, v in json.loads(self.class_names)}
        args = {'root_dir': model_dir,
                'detector_pk': self.pk,
                'class_names':{i: k for k, i in class_names.items()}
                }
        return args

    def get_class_dist(self):
        return json.loads(self.class_distribution) if self.class_distribution.strip() else {}

    def download(self):
        root_dir = settings.MEDIA_ROOT
        model_type_dir = "{}/models/".format(root_dir)
        if not os.path.isdir(model_type_dir):
            os.mkdir(model_type_dir)
        model_dir = "{}/models/{}".format(root_dir, self.uuid)
        if not os.path.isdir(model_dir):
            try:
                os.mkdir(model_dir)
            except:
                pass
        for m in self.files:
            dlpath = "{}/{}".format(model_dir,m['filename'])
            if m['url'].startswith('/'):
                shutil.copy(m['url'], dlpath)
            else:
                fs.get_path_to_file(m['url'],dlpath)
            if settings.DISABLE_NFS and sys.platform != 'darwin':
                fs.upload_file_to_remote("/models/{}/{}".format(self.uuid,m['filename']))
        if self.model_type == TrainedModel.DETECTOR and self.detector_type == TrainedModel.YOLO:
            source_zip = "{}/models/{}/model.zip".format(settings.MEDIA_ROOT, self.uuid)
            zipf = zipfile.ZipFile(source_zip, 'r')
            zipf.extractall("{}/models/{}/".format(settings.MEDIA_ROOT, self.uuid))
            zipf.close()
            os.remove(source_zip)
            self.save()
        elif self.model_type == self.INDEXER:
            dr, dcreated = Retriever.objects.get_or_create(name=self.name,source_filters={},
                                                           algorithm=Retriever.EXACT,
                                                           indexer_shasum=self.shasum)
            if dcreated:
                dr.last_built = timezone.now()
                dr.save()
        elif self.model_type == self.APPROXIMATOR:
            algo = Retriever.LOPQ if self.algorithm == 'LOPQ' else Retriever.EXACT
            dr, dcreated = Retriever.objects.get_or_create(name=self.name,
                                                           source_filters={},
                                                           algorithm=algo,
                                                           approximator_shasum=self.shasum,
                                                           indexer_shasum=self.arguments['indexer_shasum'])
            if dcreated:
                dr.last_built = timezone.now()
                dr.save()

    def ensure(self):
        for m in self.files:
            dlpath = "{}/models/{}/{}".format(settings.MEDIA_ROOT, self.uuid, m['filename'])
            if not os.path.isfile(dlpath):
                fs.ensure("/models/{}/{}".format(self.uuid,m['filename']))


class Retriever(models.Model):
    """
    Here Exact is an L2 Flat retriever
    """
    EXACT = 'E'
    LOPQ = 'L'
    MODES = (
        (LOPQ, 'LOPQ'),
        (EXACT, 'Exact'),
    )
    algorithm = models.CharField(max_length=1,choices=MODES,db_index=True,default=EXACT)
    name = models.CharField(max_length=200,default="")
    indexer_shasum = models.CharField(max_length=40,null=True)
    approximator_shasum = models.CharField(max_length=40,null=True)
    source_filters = JSONField()
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)


class Frame(models.Model):
    video = models.ForeignKey(Video, verbose_name="视频")
    event = models.ForeignKey(TEvent,null=True)
    frame_index = models.IntegerField(verbose_name="帧索引")
    name = models.CharField(max_length=200,null=True, verbose_name="名称")
    subdir = models.TextField(default="", verbose_name="子目录") # Retains information if the source is a dataset for labeling
    h = models.IntegerField(default=0, verbose_name="高度")
    w = models.IntegerField(default=0, verbose_name="宽度")
    t = models.FloatField(null=True, verbose_name="时长（秒）") # time in seconds for keyframes
    keyframe = models.BooleanField(default=False, verbose_name="关键帧") # is this a key frame for a video?
    segment_index = models.IntegerField(null=True, verbose_name="段索引")

    class Meta:
        unique_together = (("video", "frame_index"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.video_id, self.frame_index)

    def path(self,media_root=None):
        if not (media_root is None):
            return "{}/{}/frames/{}.jpg".format(media_root, self.video_id, self.frame_index)
        else:
            return "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,self.video_id,self.frame_index)

    def original_path(self):
        return self.name


class Segment(models.Model):
    """
    A video segment useful for parallel dense decoding+processing as well as streaming
    """
    video = models.ForeignKey(Video, verbose_name="视频")
    segment_index = models.IntegerField(verbose_name="段索引")
    start_time = models.FloatField(default=0.0, verbose_name="开始时间")
    end_time = models.FloatField(default=0.0, verbose_name="结束时间")
    event = models.ForeignKey(TEvent,null=True)
    metadata = models.TextField(default="{}", verbose_name="元数据")
    frame_count = models.IntegerField(default=0, verbose_name="帧数量")
    start_index = models.IntegerField(default=0, verbose_name="开始索引")
    start_frame = models.ForeignKey(Frame,null=True,related_name="segment_start", verbose_name="开始帧")
    end_frame = models.ForeignKey(Frame, null=True,related_name="segment_end", verbose_name="结束帧")

    class Meta:
        unique_together = (("video", "segment_index"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.video_id, self.segment_index)

    def path(self, media_root=None):
        if not (media_root is None):
            return "{}/{}/segments/{}.mp4".format(media_root, self.video_id, self.segment_index)
        else:
            return "{}/{}/segments/{}.mp4".format(settings.MEDIA_ROOT, self.video_id, self.segment_index)

    def framelist_path(self, media_root=None):
        if not (media_root is None):
            return "{}/{}/segments/{}.txt".format(media_root, self.video_id, self.segment_index)
        else:
            return "{}/{}/segments/{}.txt".format(settings.MEDIA_ROOT, self.video_id, self.segment_index)


class Region(models.Model):
    """
    Any 2D region over an image.
    Detections & Transforms have an associated image data.
    """
    ANNOTATION = 'A'
    DETECTION = 'D'
    SEGMENTATION = 'S'
    TRANSFORM = 'T'
    POLYGON = 'P'
    REGION_TYPES = (
        (ANNOTATION, '标注'),
        (DETECTION, '检测'),
        (POLYGON, 'Polygon'),
        (SEGMENTATION, 'Segmentation'),
        (TRANSFORM, 'Transform'),
    )
    region_type = models.CharField(max_length=1,choices=REGION_TYPES,db_index=True, verbose_name="区域类型")
    video = models.ForeignKey(Video, verbose_name="视频")
    user = models.ForeignKey(User,null=True, verbose_name="用户")
    frame = models.ForeignKey(Frame,null=True, verbose_name="帧")
    event = models.ForeignKey(TEvent, null=True)  # TEvent that created this region
    frame_index = models.IntegerField(default=-1, verbose_name="帧索引")
    segment_index = models.IntegerField(default=-1,null=True, verbose_name="段索引")
    text = models.TextField(default="", verbose_name="文本")
    metadata = JSONField(blank=True,null=True, verbose_name="元数据")
    full_frame = models.BooleanField(default=False, verbose_name="全帧")
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    polygon_points = JSONField(blank=True,null=True, verbose_name="多边形拐点")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    object_name = models.CharField(max_length=100, verbose_name="对象名称")
    confidence = models.FloatField(default=0.0, verbose_name="置信度")
    materialized = models.BooleanField(default=False)
    png = models.BooleanField(default=False, verbose_name="png格式")

    def clean(self):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index
        super(Region, self).save(*args, **kwargs)

    def path(self,media_root=None,temp_root=None):
        if temp_root:
            return "{}/{}_{}.jpg".format(temp_root, self.video_id, self.pk)
        elif not (media_root is None):
            return "{}/{}/regions/{}.jpg".format(media_root, self.video_id, self.pk)
        else:
            return "{}/{}/regions/{}.jpg".format(settings.MEDIA_ROOT, self.video_id, self.pk)

    def frame_path(self,media_root=None):
        if not (media_root is None):
            return "{}/{}/frames/{}.jpg".format(media_root, self.video_id, self.frame_index)
        else:
            return "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT, self.video_id, self.frame_index)


class QueryRegion(models.Model):
    """
    Any 2D region over a query image.
    """
    ANNOTATION = 'A'
    DETECTION = 'D'
    SEGMENTATION = 'S'
    TRANSFORM = 'T'
    POLYGON = 'P'
    REGION_TYPES = (
        (ANNOTATION, '标注'),
        (DETECTION, '检测'),
        (POLYGON, '多边形'),
        (SEGMENTATION, '分割'),
        (TRANSFORM, '变换'),
    )
    region_type = models.CharField(max_length=1,choices=REGION_TYPES,db_index=True, verbose_name="区域类型")
    query = models.ForeignKey(DVAPQL, verbose_name="查询")
    event = models.ForeignKey(TEvent, null=True)  # TEvent that created this region
    text = models.TextField(default="", verbose_name="文本")
    metadata = JSONField(blank=True,null=True, verbose_name="元数据")
    full_frame = models.BooleanField(default=False, verbose_name="全帧")
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    polygon_points = JSONField(blank=True,null=True, verbose_name="多边形拐点")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    object_name = models.CharField(max_length=100, verbose_name="对象名称")
    confidence = models.FloatField(default=0.0, verbose_name="置信度")
    png = models.BooleanField(default=False, verbose_name="png格式")


class QueryResults(models.Model):
    query = models.ForeignKey(DVAPQL, verbose_name="查询")
    retrieval_event = models.ForeignKey(TEvent,null=True)
    video = models.ForeignKey(Video, verbose_name="视频")
    frame = models.ForeignKey(Frame, verbose_name="帧")
    detection = models.ForeignKey(Region,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100, verbose_name="算法")
    distance = models.FloatField(default=0.0)


class QueryRegionResults(models.Model):
    query = models.ForeignKey(DVAPQL)
    query_region = models.ForeignKey(QueryRegion, verbose_name="被查询区域")
    retrieval_event = models.ForeignKey(TEvent,null=True)
    video = models.ForeignKey(Video, verbose_name="视频")
    frame = models.ForeignKey(Frame, verbose_name="帧")
    detection = models.ForeignKey(Region,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100, verbose_name="算法")
    distance = models.FloatField(default=0.0)


class IndexEntries(models.Model):
    video = models.ForeignKey(Video, verbose_name="视频")
    features_file_name = models.CharField(max_length=100, verbose_name="特征文件名称")
    entries_file_name = models.CharField(max_length=100, verbose_name="条目文件名称")
    algorithm = models.CharField(max_length=100, verbose_name="算法")
    indexer = models.ForeignKey(TrainedModel, null=True, verbose_name="索引器")
    indexer_shasum = models.CharField(max_length=40, verbose_name="索引器 shasum")
    approximator_shasum = models.CharField(max_length=40, null=True, verbose_name="相似器 shasum")
    detection_name = models.CharField(max_length=100, verbose_name="索引对象")
    count = models.IntegerField(verbose_name="数量")
    approximate = models.BooleanField(default=False, verbose_name="相似")
    contains_frames = models.BooleanField(default=False, verbose_name="包含帧")
    contains_detections = models.BooleanField(default=False, verbose_name="包含检测")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    event = models.ForeignKey(TEvent, null=True)

    class Meta:
        unique_together = ('video', 'entries_file_name',)

    def __unicode__(self):
        return "{} in {} index by {}".format(self.detection_name, self.algorithm, self.video.name)

    def npy_path(self, media_root=None):
        if not (media_root is None):
            return "{}/{}/indexes/{}".format(media_root, self.video_id, self.features_file_name)
        else:
            return "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, self.video_id, self.features_file_name)

    def entries_path(self, media_root=None):
        if not (media_root is None):
            return "{}/{}/indexes/{}".format(media_root, self.video_id, self.entries_file_name)
        else:
            return "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, self.video_id, self.entries_file_name)

    def load_index(self,media_root=None):
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        video_dir = "{}/{}".format(media_root, self.video_id)
        if not os.path.isdir(video_dir):
            os.mkdir(video_dir)
        index_dir = "{}/{}/indexes".format(media_root, self.video_id)
        if not os.path.isdir(index_dir):
            os.mkdir(index_dir)
        dirnames = {}
        if self.features_file_name.strip():
            fs.ensure(self.npy_path(media_root=''), dirnames, media_root)
            vectors = np.load(self.npy_path(media_root))
        else:
            vectors = None
        fs.ensure(self.entries_path(media_root=''),dirnames,media_root)
        entries = json.load(file(self.entries_path(media_root)))
        return vectors,entries


class Tube(models.Model):
    """
    A tube is a collection of sequential frames / regions that track a certain object
    or describe a specific scene
    """
    video = models.ForeignKey(Video,null=True, verbose_name="视频")
    frame_level = models.BooleanField(default=False, verbose_name="帧级别")
    start_frame_index = models.IntegerField(verbose_name="开始索引")
    end_frame_index = models.IntegerField(verbose_name="结束索引")
    start_frame = models.ForeignKey(Frame,null=True,related_name="start_frame", verbose_name="开始帧")
    end_frame = models.ForeignKey(Frame,null=True,related_name="end_frame", verbose_name="结束帧")
    start_region = models.ForeignKey(Region,null=True,related_name="start_region", verbose_name="开始区域")
    end_region = models.ForeignKey(Region,null=True,related_name="end_region", verbose_name="结束区域")
    text = models.TextField(default="", verbose_name="文本")
    metadata = JSONField(blank=True,null=True, verbose_name="元数据")
    source = models.ForeignKey(TEvent,null=True, verbose_name="资源")


class Label(models.Model):
    name = models.CharField(max_length=200, verbose_name="名称")
    set = models.CharField(max_length=200,default="", verbose_name="集合名称")
    metadata = JSONField(blank=True,null=True, verbose_name="元数据")
    text = models.TextField(null=True,blank=True, verbose_name="文本")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)

    class Meta:
        unique_together = (("name", "set"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.name, self.set)


class FrameLabel(models.Model):
    video = models.ForeignKey(Video,null=True, verbose_name="视频")
    frame_index = models.IntegerField(default=-1, verbose_name="帧索引")
    segment_index = models.IntegerField(null=True, verbose_name="段索引")
    frame = models.ForeignKey(Frame, verbose_name="帧")
    label = models.ForeignKey(Label, verbose_name="标签")
    event = models.ForeignKey(TEvent,null=True)

    def clean(self):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index
        super(FrameLabel, self).save(*args, **kwargs)


class RegionLabel(models.Model):
    video = models.ForeignKey(Video,null=True, verbose_name="视频")
    frame = models.ForeignKey(Frame,null=True, verbose_name="帧")
    frame_index = models.IntegerField(default=-1, verbose_name="帧索引")
    segment_index = models.IntegerField(null=True, verbose_name="段索引")
    region = models.ForeignKey(Region, verbose_name="区域")
    label = models.ForeignKey(Label, verbose_name="标签")
    event = models.ForeignKey(TEvent,null=True)

    def clean(self):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index
        super(RegionLabel, self).save(*args, **kwargs)


class SegmentLabel(models.Model):
    video = models.ForeignKey(Video,null=True, verbose_name="视频")
    segment_index = models.IntegerField(default=-1, verbose_name="段索引")
    segment = models.ForeignKey(Segment, verbose_name="段")
    label = models.ForeignKey(Label, verbose_name="标签")
    event = models.ForeignKey(TEvent, null=True)

    def clean(self):
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.segment.segment_index

    def save(self, *args, **kwargs):
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.segment.segment_index
        super(SegmentLabel, self).save(*args, **kwargs)


class TubeLabel(models.Model):
    video = models.ForeignKey(Video,null=True, verbose_name="视频")
    tube = models.ForeignKey(Tube)
    label = models.ForeignKey(Label, verbose_name="标签")
    event = models.ForeignKey(TEvent, null=True)


class VideoLabel(models.Model):
    video = models.ForeignKey(Video, verbose_name="视频")
    label = models.ForeignKey(Label, verbose_name="标签")
    event = models.ForeignKey(TEvent, null=True)


class DeletedVideo(models.Model):
    name = models.CharField(max_length=500,default="", verbose_name="名称")
    description = models.TextField(default="", verbose_name="描述")
    uploader = models.ForeignKey(User,null=True,related_name="user_uploader", verbose_name="上传人")
    url = models.TextField(default="", verbose_name="网址")
    deleter = models.ForeignKey(User,related_name="user_deleter",null=True, verbose_name="删除人")
    original_pk = models.IntegerField(verbose_name="原始主键")

    def __unicode__(self):
        return u'Deleted {}'.format(self.name)


class ManagementAction(models.Model):
    parent_task = models.CharField(max_length=500, default="", verbose_name="父任务")
    op = models.CharField(max_length=500, default="")
    host = models.CharField(max_length=500, default="", verbose_name="主机")
    message = models.TextField(verbose_name="信息")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    ping_index = models.IntegerField(null=True)


class SystemState(models.Model):
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    tasks = models.IntegerField(default=0, verbose_name="任务数量")
    pending_tasks = models.IntegerField(default=0, verbose_name="已暂停任务数量")
    completed_tasks = models.IntegerField(default=0,verbose_name="已完成任务数量")
    processes = models.IntegerField(default=0, verbose_name="进程数量")
    pending_processes = models.IntegerField(default=0, verbose_name="已暂停进程数量")
    completed_processes = models.IntegerField(default=0, verbose_name="已完成进程数量")
    queues = JSONField(blank=True,null=True, verbose_name="队列")
    hosts = JSONField(blank=True,null=True, verbose_name="主机")


class QueryRegionIndexVector(models.Model):
    event = models.ForeignKey(TEvent)
    query_region = models.ForeignKey(QueryRegion, verbose_name="被查询区域")
    vector = models.BinaryField()
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)


class TrainingSet(models.Model):
    DETECTION = 'D'
    INDEXING = 'I'
    LOPQINDEX = 'A'
    CLASSIFICATION = 'C'
    TRAIN_TASK_TYPES = (
        (DETECTION, '检测'),
        (INDEXING, '索引'),
        (CLASSIFICATION, '分类')
    )
    IMAGES = 'I'
    VIDEOS = 'V'
    INSTANCE_TYPES = (
        (IMAGES, '图像'),
        (VIDEOS, '视频'),
    )
    event = models.ForeignKey(TEvent)
    training_task_type = models.CharField(max_length=1,choices=TRAIN_TASK_TYPES,db_index=True,default=DETECTION, verbose_name="训练任务类型")
    instance_type = models.CharField(max_length=1,choices=INSTANCE_TYPES,db_index=True,default=IMAGES, verbose_name="实例类型")
    count = models.IntegerField(null=True, verbose_name="数量")
    name = models.CharField(max_length=500,default="", verbose_name="名称")
    built = models.BooleanField(default=False, verbose_name="已创建")
    created = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
