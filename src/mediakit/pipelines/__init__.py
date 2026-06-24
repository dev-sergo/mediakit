from mediakit.pipelines.article_cover import ArticleCoverPipeline
from mediakit.pipelines.photo_animate import PhotoAnimatePipeline
from mediakit.pipelines.photo_finalize import PhotoFinalizePipeline
from mediakit.pipelines.product_shot import ProductShotPipeline
from mediakit.pipelines.responsive_set import ResponsiveSetPipeline
from mediakit.pipelines.seamless_video import SeamlessVideoPipeline
from mediakit.pipelines.txt_to_video_hq import TxtToVideoHqPipeline

article_cover = ArticleCoverPipeline()
photo_animate = PhotoAnimatePipeline()
photo_finalize = PhotoFinalizePipeline()
product_shot = ProductShotPipeline()
responsive_set = ResponsiveSetPipeline()
seamless_video = SeamlessVideoPipeline()
txt_to_video_hq = TxtToVideoHqPipeline()

__all__ = [
    "article_cover",
    "photo_animate",
    "photo_finalize",
    "product_shot",
    "responsive_set",
    "seamless_video",
    "txt_to_video_hq",
]
