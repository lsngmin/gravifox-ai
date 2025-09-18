"""
MINTIME builder stub.

Fill in `build()` to construct your Extractor(TimeSformer backbone) and
Classifier(head) modules, then the runner/exporter will load your state_dict
checkpoints into them.

Example (pseudo):

    import torch.nn as nn
    from your_pkg.timesformer import TimeSformerBackbone
    from your_pkg.heads import BinaryHead

    def build():
        extractor = TimeSformerBackbone(img_size=224, patch_size=16, depth=12, ...)
        classifier = BinaryHead(in_features=extractor.out_dim, num_classes=2)
        return extractor, classifier

Return:
    (extractor_module, classifier_module)
"""

def build():
    raise NotImplementedError(
        "Implement mintime_builder.build() to return (extractor, classifier)."
    )

